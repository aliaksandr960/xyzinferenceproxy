import threading
import queue
import time
import collections
import requests
import numpy as np
import cv2
import torch
from fastapi import FastAPI
from fastapi.responses import Response


def add_text_default(image, text, position):
    font_scale = 1.0
    font_color = (255,255,255)
    thickness = 2
    lineType = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, position, font,  font_scale, font_color, thickness, lineType)


def add_tile_info(image, x, y, z, tz=''):
    add_text_default(image, "x={}".format(str(x)),(16, 64))
    add_text_default(image, "y={}".format(str(y)), (16, 128))
    add_text_default(image, "z={}/{}".format(str(z), str(tz)),(16, 196))


def get_cache_index(x, y, z):
    return "{}_{}_{}".format(str(x), str(y), str(z))


def download_xyz_tile(template, x, y, z, brg_convesion=True,
                request_timeout=2, retry_timeout=2, attempts=2):
    while True:
        try:
            r = requests.get(template.format(x=x, y=y, z=z), timeout=request_timeout)
            if r.status_code != 200:
                raise ValueError('XYZ response error, x={}, y={}, z={}, code={}'.format(str(x), str(y), str(z), str(r.status_code)))
            else:
                break
        except:
            attempts -= 1
            if attempts == 0:
                raise
            else:
                time.sleep(retry_timeout)

    image_array = np.frombuffer(r.content, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if brg_convesion:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_tiles(xyz_list, tile_downloader, tile_cache):
    result = []
    for x, y, z in xyz_list:
        xyz_index = get_cache_index(x, y, z)
        if xyz_index not in tile_cache:
            done_event = threading.Event()
            try:
                tile_cache[xyz_index] = (None, done_event)
                tile_image = tile_downloader(x, y, z)
                tile_cache[xyz_index] = (tile_image, done_event)
            finally:
                # There could be another function in parallel, that wait for this tile
                # And we must to set event and let another function to crash, too
                done_event.set()
        _, done = tile_cache[xyz_index]
        done.wait()
        image, _ = tile_cache[xyz_index]
        result.append(image)
    return result


def get_tile_mosaic(x, y, z, tile_list_downloader, tile_size=(256, 256, 3), pad=1):
    th, tw, tc = tile_size
    mh, mw, mc = (pad *2 +1) * th, (pad *2 +1) * tw, tc
    mosaic = np.zeros([mh, mw, mc], dtype=np.uint8)

    xs = [x + i for i in range(-pad, pad+1)]
    ys = [y + i for i in range(-pad, pad+1)]
    xyz_list = []
    for j in ys:
        for i in xs:
            xyz_list.append((i, j, z))

    tiles = tile_list_downloader(xyz_list)
    for i in range(pad *2 +1):
        for j in range(pad *2 +1):
            sh, sw = i *th, j *tw
            eh, ew = (i +1) *th, (j +1) *tw
            tile = tiles.pop(0)
            mosaic[sh:eh, sw:ew, :] = tile
    
    return mosaic


def do_inference(image, model, channel, device):
    # Add batch-dim, hwc -> chw
    image_t = torch.unsqueeze(torch.from_numpy(np.moveaxis(image, 2, 0)), dim=0).to(device)
    # Remove batch-dim, extract class channel
    prediction = torch.squeeze(model(image_t)[:, channel, :, :]).contiguous()
    # Comment, if you apply activation function inside your model
    prediction = prediction.sigmoid()
    # Device -> CPU -> Numpy
    prediction_np = prediction.detach().cpu().numpy()
    # 0-1 range (default in most models) -> 0-255 range
    prediction_np *= 255
    # Float -> Unit8
    prediction_np = prediction_np.astype(np.uint8)
    return prediction_np


def do_inference_on_queue(inferenc_f, q, m):
    while True:
        item, index, event = q.get()
        prediction = inferenc_f(item)
        m[index] = prediction
        event.set()


## ---> Confiuration and run

# XYZ source configuration
mapbox_template = 'https://a.tiles.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}.jpg?access_token=pk.eyJ1Ijoib3BlbnN0cmVldG1hcCIsImEiOiJja2w5YWt5bnYwNjZmMnFwZjhtbHk1MnA1In0.eq2aumBK6JuRoIuBMm6Gew'
target_z = 18
tile_size = (256, 256, 3)
th, tw, tc = tile_size

# Cache configuration
max_cached_tiles = 1024
tile_cache = collections.OrderedDict()

# Tile donwnloader cofiguration and init
tile_downloader = lambda x, y, z: download_xyz_tile(mapbox_template, x, y, z, brg_convesion=False,
                                                    request_timeout=2, retry_timeout=2, attempts=2)
tile_list_downloader = lambda xyz_list: get_tiles(xyz_list, tile_downloader, tile_cache)

# Model cofiguration and init
device = 'cuda:0'
model = torch.jit.load('path.to.model.jit.pt', map_location=device)
model.eval()

# Inference thread and queues configuration and init
inference_f = lambda image: do_inference(image, model, channel=1, device=device)
inference_q = queue.Queue()
inference_m = dict()
threading.Thread(target=lambda: do_inference_on_queue(inference_f, inference_q, inference_m), daemon=False).start()

# Fast API init
app = FastAPI()

@app.get("/{x}/{y}/{z}.png", responses = {200: {"content": {"image/png": {}}}}, response_class=Response,)
def get_tile(x, y, z):
    x, y, z = int(x), int(y), int(z)
    if z == target_z:
        # If there is target Zoom -> do inference
        mosaic = get_tile_mosaic(x, y, z, tile_list_downloader, tile_size=tile_size)
        inference_i = '{}_{}'.format(get_cache_index(x, y, z), str(time.time()))
        inference_e = threading.Event()
        inference_q.put((mosaic, inference_i, inference_e))
        inference_e.wait()

        prediction = inference_m.pop(inference_i)
        prediction_crop = prediction[th:th *2, tw:tw *2]
        tile = np.stack([prediction_crop, prediction_crop, prediction_crop], axis=-1)
    else:
        # If not -> just pass tile with info
        tile = tile_downloader(x, y, z)
        add_tile_info(tile, x, y, z, target_z)
    
    # If limit reached -> clear cache
    while len(tile_cache) > max_cached_tiles:
        tile_cache.popitem()
    
    # Return PNG tile as response
    tile_as_bytes = cv2.imencode(".png", tile)[1].tobytes()
    return Response(content=tile_as_bytes, media_type="image/png")
