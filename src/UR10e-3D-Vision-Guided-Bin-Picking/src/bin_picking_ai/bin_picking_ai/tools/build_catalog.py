#!/usr/bin/env python3
"""Build ``data/catalog.json``: size + colour signature + spoken names of every pickable model.

Usage: python3 build_catalog.py <gazebo_pkg_dir> <out.json>
Needs: numpy, trimesh, opencv, pycollada not required (OBJ meshes only).
"""
import json
import os
import re
import sys

import cv2
import numpy as np
import trimesh

# model dir -> (short id, spoken names / aliases, category words)
NAMES = {
    'Weston_No_33_Signature_Sausage_Tonic_12_fl_oz': ('sausage_tonic', ['sausage tonic', 'weston', 'weston no 33', 'tonic', 'brown bottle'], ['bottle']),
    'PottedMeatCan': ('potted_meat_can', ['potted meat', 'spam', 'meat can', 'can of meat', 'tin'], ['can']),
    'MustardBottle': ('mustard_bottle', ['mustard', 'mustard bottle', 'french mustard'], ['bottle']),
    'YumYum_D3_Liquid': ('vitamin_d3', ['vitamin d', 'vitamin d3', 'yum yum', 'yumyum', 'd3', 'drops', 'orange box'], ['box']),
    'Wilton_PreCut_Parchment_Sheets_10_x_15_24_sheets': ('parchment', ['parchment', 'parchment sheets', 'baking paper', 'wilton parchment'], ['box']),
    'Wilton_Pearlized_Sugar_Sprinkles_525_oz_Gold': ('sprinkles', ['sprinkles', 'sugar sprinkles', 'wilton sprinkles', 'sugar'], ['jar']),
    'Whey_Protein_Vanilla_12_Packets': ('whey_protein', ['whey', 'whey protein', 'protein', 'protein packets'], ['box']),
    'Nescafe_Momento_Mocha_Specialty_Coffee_Mix_8_ct': ('coffee_mix', ['coffee', 'nescafe', 'mocha', 'coffee mix'], ['box']),
    'Orbit_Bubblemint_Mini_Bottle_6_ct': ('chewing_gum', ['orbit', 'gum', 'chewing gum', 'bubblemint', 'pink box'], ['box']),
    'Nestle_Nesquik_Chocolate_Powder_Flavored_Milk_Additive_109_Oz_Canister': ('nesquik', ['nesquik', 'chocolate powder', 'chocolate', 'nestle', 'cocoa'], ['canister']),
    'Vans_Cereal_Honey_Nut_Crunch_11_oz_box': ('cereal', ['cereal', 'cereal box', 'vans', 'honey crunch', 'honey nut'], ['box']),
    'Marc_Anthony_True_Professional_Strictly_Curls_Curl_Defining_Lotion': ('curl_lotion', ['lotion', 'curl lotion', 'marc anthony', 'curls', 'hair lotion', 'curl defining lotion'], ['bottle']),
    'Twinlab_Nitric_Fuel': ('nitric_fuel', ['nitric fuel', 'twinlab', 'pre workout', 'supplement'], ['pouch']),
}


def texture_signature(model_dir):
    pngs = []
    for root, _, files in os.walk(model_dir):
        pngs += [os.path.join(root, f) for f in files if f.lower().endswith('.png')]
    if not pngs:
        return None
    im = cv2.imread(pngs[0])
    im = cv2.resize(im, (512, 512), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    used = hsv[..., 2] > 40                       # drop the black/unused atlas area
    if used.sum() < 100:
        used = np.ones_like(used)
    h = cv2.calcHist([hsv], [0], used.astype(np.uint8), [18], [0, 180]).ravel()
    s = cv2.calcHist([hsv], [1], used.astype(np.uint8), [4], [0, 256]).ravel()
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)[used].mean(0)
    return dict(mean_rgb=[round(float(x), 1) for x in rgb],
                hue_hist=[round(float(x), 4) for x in (h / max(h.sum(), 1))],
                sat_hist=[round(float(x), 4) for x in (s / max(s.sum(), 1))])


def main(argv=None):
    argv = argv or sys.argv[1:]
    gz_dir, out = argv
    models_dir = os.path.join(gz_dir, 'models')
    world = open(os.path.join(gz_dir, 'worlds', 'bin_picking_world.sdf'), encoding='utf-8').read()
    cat = {}
    for model, (cid, aliases, cats) in NAMES.items():
        mdir = os.path.join(models_dir, model)
        sdf = open(os.path.join(mdir, 'model.sdf'), encoding='utf-8').read()
        uris = re.findall(r'<uri>([^<]+)</uri>', sdf)
        obj = ([u for u in uris if u.endswith('.obj')] or [u.replace('.dae', '.obj') for u in uris if u.endswith('.dae')])[0]
        mesh = trimesh.load(os.path.join(mdir, obj), force='mesh', process=False)
        dims = (mesh.bounds[1] - mesh.bounds[0]).tolist()
        entities = re.findall(r'model://%s</uri>\s*<name>(\w+)</name>' % model, world)
        cat[cid] = dict(model=model, entities=entities, dims_xyz=[round(d, 4) for d in dims],
                        names=[cid.replace('_', ' ')] + aliases, categories=cats,
                        appearance=texture_signature(mdir))
        print(f'{cid:16s} dims={np.round(dims, 3)} entities={entities}')
    json.dump(cat, open(out, 'w'), indent=1)
    print('wrote', out)


if __name__ == '__main__':
    main()
