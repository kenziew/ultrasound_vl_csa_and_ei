from qtpy.QtWidgets import QLabel
from natsort import natsorted
from magicgui import magicgui
from utils.vl_utils import *
from utils.utils import *
import napari
import glob
import json
import time
import cv2
import os

def update_index_label(current_index: int, list_length: int):
    """Update widget dislaying index of current ultrasound viewed in Napari GUI"""
    global index_label
    index_label.setText(f"Image {current_index+1} of {list_length}")

def add_polygon_to_mask_layer(polygons: list):
    """Add polygon(s) to mask layer in Napari GUI"""
    global mask_layer
    labels = ["Vastus Lateralis", "Subcutaneous Tissue"]
    if mask_layer is None:
        mask_layer = viewer.add_shapes(
            polygons,
            shape_type='polygon',
            edge_color=['red', 'blue'],
            face_color='transparent',
            properties={'label': labels},
            text={'string': '{label}', 'size': 12, 'color': 'white', 'anchor': 'center'},
            edge_width=5,
            name="VL Predictions"
        )
    else:
        mask_layer.selected_data = set(range(len(mask_layer.data)))
        mask_layer.remove_selected()

        mask_layer.add(polygons, 
            shape_type='polygon',
            edge_color=['red', 'blue'],
            face_color='transparent'
        )
        mask_layer.properties = {'label': labels}
        mask_layer.text = {'string': '{label}', 'size': 12, 'color': 'white', 'anchor': 'center'}
        mask_layer.edge_width = 5

@magicgui(call_button="Load Folder",
          image_dir={"label": "Image Folder:"})
def load_folder(image_dir: str):
    """Retrieve all ultrasound images from a folder."""
    global cfg, images, image_layer, mask_layer
    cfg['image_folder'] = image_dir
    images = glob.glob(os.path.join(image_dir, '**', '*'), recursive=True)
    # images = [str(image) for image in images if image.split('.')[-1] in exts]
    images = natsorted(images)
    if not images:
        msg = f"No Images Found In: {image_dir}"
        popup_message(msg)
        return

    make_subdirs(image_dir, cfg['output_path'], cfg['output_mask_path'], images)

    if os.path.exists(cfg['index_json_path']):
        with open(cfg['index_json_path']) as f:
            data = json.load(f)

        cfg['current_index'] = data["current_index"]
        msg = f"Detected Previous Usage of the Model. Loading Current Index from JSON File at: {cfg['index_json_path']}"
        popup_message(msg)
    else:
        cfg['current_index'] = 0
        data = {"current_index": 0}
        with open(cfg['index_json_path'], 'w') as f:
            json.dump(data, f, indent=4)
        msg = f"No Previous Usage of the Model Detected. Created JSON File to Store Indexes at: {cfg['index_json_path']}"
        popup_message(msg)

    img, cm_per_pixel, cm2_per_pixel = load_dicom(images[cfg['current_index']])
    cfg['cm_per_pixel'] = cm_per_pixel
    cfg['cm2_per_pixel'] = cm2_per_pixel
    if image_layer is None:
        image_layer = viewer.add_image(img, name="Ultrasound")
        update_index_label(cfg['current_index'], len(images))
    else:
        image_layer.data = img
        update_index_label(cfg['current_index'], len(images))
    run_model()

@magicgui(call_button="Set Conversion",
    manual_depth={"widget_type": "CheckBox", "label": "Manual Pixel Conversion?"},
    cm_per_pixel={"widget_type": "FloatSpinBox", "min": 0, "max": 1000, "enabled": False, "label": "CM Per Pixel:"},
    cm2_per_pixel={"widget_type": "FloatSpinBox", "min": 0, "max": 1000, "enabled": False, "label": "CM^2 Per Pixel:"}
)
def set_depth(manual_depth: bool=False, cm_per_pixel: float=0.0, cm2_per_pixel: float=0.0):
    if manual_depth:
        if cm_per_pixel == 0.0:
            return
        cfg['cm_per_pixel'] = cm_per_pixel
        cfg['cm2_per_pixel'] = cm2_per_pixel
        popup_message(f"Set cm per pixel to {cm_per_pixel} and cm^2 per pixel to {cm2_per_pixel}")
    else:
        cfg['cm_per_pixel'] = None
        cfg['cm2_per_pixel'] = None
        set_depth.cm_per_pixel.value = 0.0
        set_depth.cm2_per_pixel.value = 0.0

@set_depth.manual_depth.changed.connect
def _toggle_depth_checkbox(e):
    set_depth.cm_per_pixel.enabled = e
    set_depth.cm2_per_pixel.enabled = e

set_depth.manual_depth.changed.connect(lambda _: set_depth())

@magicgui(call_button="Run Model")
def run_model():
    """Run AI model on the current image."""
    global cfg, image_layer, mask_layer, models, images

    if not images:
        msg = "No Images Loaded!"
        popup_message(msg)
        return

    img = image_layer.data
    vl_pred, subq_pred = get_mask(cfg, img, models)
    vl_polygon = mask_to_polygon(vl_pred)
    subq_polygon = mask_to_polygon(subq_pred)
    polygons = [vl_polygon[0], subq_polygon[0]]
    add_polygon_to_mask_layer(polygons)

@magicgui(call_button="Next Image")
def next_image():
    """Go to next image in folder."""
    global cfg, images, image_layer, mask_layer, set_depth
    if not images:
        return
    save_corrections()
    calculate()
    cfg['current_index'] = (cfg['current_index'] + 1) % len(images)
    if cfg['current_index'] == 0:
        msg = "Last Image in Folder Has Been Analyzed."
        popup_message(msg)
    save_current_index(cfg['current_index'], cfg['index_json_path'])
    update_index_label(cfg['current_index'], len(images))
    set_depth.manual_depth.value = False

    img, cm_per_pixel, cm2_per_pixel = load_dicom(images[cfg['current_index']])
    image_layer.data = img
    cfg['cm_per_pixel'] = cm_per_pixel
    cfg['cm2_per_pixel'] = cm2_per_pixel

    rel_path = os.path.relpath(images[cfg['current_index']], start=cfg['image_folder'])
    vl_poly_path = os.path.join(cfg['output_poly_path'], os.path.splitext(rel_path)[0] + '_vl_poly.json')
    subq_poly_path = os.path.join(cfg['output_poly_path'], os.path.splitext(rel_path)[0] + '_subq_poly.json')
    if os.path.exists(vl_poly_path):
        vl_poly = load_polygon(vl_poly_path)
        subq_poly = load_polygon(subq_poly_path)
        polygons = [vl_poly[0], subq_poly[0]]
        add_polygon_to_mask_layer(polygons)
    else:
        run_model()

@magicgui(call_button="Previous Image")
def prev_image():
    """Go to previous image in folder."""
    global cfg, images, image_layer, mask_layer, set_depth
    if not images:
        msg = "No Images Loaded!"
        popup_message(msg)
        return
    save_corrections()
    calculate()
    cfg['current_index'] = (cfg['current_index'] - 1) % len(images)
    save_current_index(cfg['current_index'], cfg['index_json_path'])
    update_index_label(cfg['current_index'], len(images))
    set_depth.manual_depth.value = False

    img, cm_per_pixel, cm2_per_pixel = load_dicom(images[cfg['current_index']])
    image_layer.data = img
    cfg['cm_per_pixel'] = cm_per_pixel
    cfg['cm2_per_pixel'] = cm2_per_pixel

    rel_path = os.path.relpath(images[cfg['current_index']], start=cfg['image_folder'])
    vl_poly_path = os.path.join(cfg['output_poly_path'], os.path.splitext(rel_path)[0] + '_vl_poly.json')
    subq_poly_path = os.path.join(cfg['output_poly_path'], os.path.splitext(rel_path)[0] + '_subq_poly.json')
    if os.path.exists(vl_poly_path):
        vl_poly = load_polygon(vl_poly_path)
        subq_poly = load_polygon(subq_poly_path)
        polygons = [vl_poly[0], subq_poly[0]]
        add_polygon_to_mask_layer(polygons)
    else:
        run_model()

@magicgui(call_button="Save Corrections")
def save_corrections():
    """Save corrected mask as polygon and vizualization"""
    global cfg, images, image_layer, mask_layer
    if mask_layer is None or not images:
        msg = "No Mask To Save!"
        popup_message(msg)
        return
    if not images:
        msg = "No Images Loaded!"
        popup_message(msg)
        return
    vl_poly = [mask_layer.data[0]]
    subq_poly = [mask_layer.data[1]]
    vl_mask = polygon_to_mask(vl_poly, image_layer.data.shape[:2])
    subq_mask = polygon_to_mask(subq_poly, image_layer.data.shape[:2])
    rel_path = os.path.relpath(images[cfg['current_index']], start=cfg['image_folder'])
    mask_out_path = os.path.join(cfg['output_mask_path'], os.path.splitext(rel_path)[0] + "_mask.png")
    vl_poly_out_path = os.path.join(cfg['output_poly_path'], os.path.splitext(rel_path)[0] + '_vl_poly.json')
    subq_poly_out_path = os.path.join(cfg['output_poly_path'], os.path.splitext(rel_path)[0] + '_subq_poly.json')
    
    input_img = cv2.cvtColor(image_layer.data, cv2.COLOR_RGB2BGR)
    viz = make_vl_viz(input_img, vl_mask, subq_mask)
    cv2.imwrite(str(mask_out_path), viz)
    save_polygon(vl_poly, vl_poly_out_path)
    save_polygon(subq_poly, subq_poly_out_path)

    msg = f"Saved Predictions At: {mask_out_path}"
    popup_message(msg)

@magicgui(call_button="Calculate Values")
def calculate():
    """Calculate clinical values and return popup messages if values can't be calculated"""
    global cfg, images, image_layer, mask_layer
    if np.isnan(cfg['cm_per_pixel']) or np.isnan(cfg['cm2_per_pixel']):
        msg = "Please Manually Set CM Per Pixel and CM^2 Per Pixel! They Were NOT Able to Be Read From the Dicom!"
        popup_message(msg)
        return

    calc_fail = 0
    vl_mask = polygon_to_mask([mask_layer.data[0]], image_layer.data.shape[:2])
    subq_mask = polygon_to_mask([mask_layer.data[1]], image_layer.data.shape[:2])

    vl_csa, csa_fail = calculate_vl_csa(cfg, vl_mask)
    if csa_fail:
        msg = "Vastus Lateralis Cross-Sectional Area Calculation Failed!"
        popup_message(msg)

    vl_thickness, thickness_fail = calculate_vl_thickness(cfg, vl_mask)
    if thickness_fail:
        msg = "Vastus Lateralis Thickness Calculation Failed!"
        popup_message(msg)

    vl_std_ei, vl_avg_ei, ei_fail = calculate_vl_ei(cfg, image_layer.data, vl_mask)
    if ei_fail == 1 or ei_fail == 3:
        msg = "Vastus Lateralis Standard Deviation of Echo Intensity Calculation Failed!"
        popup_message(msg)
    if ei_fail == 2 or ei_fail == 3:
        msg = "Vastus Lateralis Average Echo Intensity Calculation Failed!"
        popup_message(msg)

    subq_thickness, subq_thickness_fail = calculate_subq_thickness(cfg, image_layer.data, subq_mask, vl_mask)
    if subq_thickness_fail == 1:
        msg = "Subcutaneous Tissue Thickness Calculation Failed!"
        popup_message(msg)

    muller_adjusted_vl_avg_ei, young_adjusted_vl_avg_ei = calculate_corrected_eis(cfg, vl_avg_ei, subq_thickness)

    if thickness_fail + csa_fail + ei_fail + subq_thickness_fail > 0:
        calc_fail = 1

    img_path = os.path.relpath(images[cfg['current_index']], start=cfg['image_folder'])
    df = pd.read_csv(cfg['calc_csv_path'])
    row = [img_path, calc_fail, cfg['cm_per_pixel'], cfg['cm2_per_pixel'], vl_thickness, vl_csa, vl_avg_ei,
            vl_std_ei, subq_thickness, muller_adjusted_vl_avg_ei, young_adjusted_vl_avg_ei, 0, 0, 0, 0, calc_fail]
    if img_path in df['img_path'].values:
        df.loc[df['img_path'] == img_path] = row
    else:
        df.loc[len(df)] = row
    df.to_csv(cfg['calc_csv_path'], index=False)

    if calc_fail:
        msg = "Calculations Saved with At Least One Failure!"
    else:
        msg = "Calculations Saved!"
    popup_message(msg)

@magicgui(call_button="Run Model on All Images")
def run_all_images():
    global cfg, images, models
    confirm = confirm_action()
    if not confirm:
        return
    
    start_time = time.time()
    analysis(cfg, images, models)
    end_time = time.time()
    msg = f"Analysis was completed after: {end_time-start_time:.2f} seconds"
    popup_message(msg)

def main():
    global viewer, image_layer, mask_layer, cfg, images, models, index_label
    viewer = napari.Viewer()
    image_layer = None
    mask_layer = None
    cfg = setup_cfg("configs/vl_gui_config.json")
    images = []
    os.makedirs(cfg['output_path'], exist_ok=True)
    os.makedirs(cfg['output_mask_path'], exist_ok=True)
    os.makedirs(cfg['output_poly_path'], exist_ok=True)
    make_calc_csv(cfg)
    models = setup_ai_models(cfg['model_directory'], cfg['device'])

    viewer.window.add_dock_widget(load_folder, area="right")
    viewer.window.add_dock_widget(set_depth, area="right")
    viewer.window.add_dock_widget(run_model, area="right")
    viewer.window.add_dock_widget(save_corrections, area="right")
    viewer.window.add_dock_widget(calculate, area="right")
    viewer.window.add_dock_widget(run_all_images, area="bottom")
    viewer.window.add_dock_widget(next_image, area="right")
    viewer.window.add_dock_widget(prev_image, area="right")
    index_label = QLabel(f"Image {cfg['current_index']+1} of {len(images)}")
    viewer.window.add_dock_widget(index_label, area="bottom")

    napari.run()

if __name__ == "__main__":
    main()