#!/usr/bin/env python3
import os
import requests
import gdown
import json


def download_checkpoint(url, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        print("download checkpoints ......")
        response = requests.get(url, stream=True)
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print("download successfully!")

    return filepath


def download_checkpoint_from_google_drive(file_id, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        print(
            "Downloading checkpoints from Google Drive... tips: If you cannot see the progress bar, please try to download it manuall \
              and put it in the checkpointes directory. E2FGVI-HQ-CVPR22.pth: https://github.com/MCG-NKU/E2FGVI(E2FGVI-HQ model)"
        )
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filepath, quiet=False)
        print("Downloaded successfully!")

    return filepath


# convert points input to prompt state
def get_prompt(click_state, click_input):
    inputs = json.loads(click_input)
    points = click_state[0]
    labels = click_state[1]
    for input in inputs:
        points.append(input[:2])
        labels.append(input[2])
    click_state[0] = points
    click_state[1] = labels
    prompt = {
        "prompt_type": ["click"],
        "input_point": click_state[0],
        "input_label": click_state[1],
        "multimask_output": "True",
    }
    return prompt


def add_multi_mask(video_state, interactive_state, mask_dropdown):
    try:
        mask = video_state["masks"][video_state["select_frame_number"]]
        interactive_state["multi_mask"]["masks"].append(mask)
        interactive_state["multi_mask"]["mask_names"].append(
            "mask_{:03d}".format(len(interactive_state["multi_mask"]["masks"]))
        )
        mask_dropdown.append(
            "mask_{:03d}".format(len(interactive_state["multi_mask"]["masks"]))
        )
        select_frame, run_status = show_mask(
            video_state, interactive_state, mask_dropdown
        )

        operation_log = [
            ("", ""),
            (
                "Added a mask, use the mask select for target tracking or inpainting.",
                "Normal",
            ),
        ]
    except:
        operation_log = [
            ("Please click the left image to generate mask.", "Error"),
            ("", ""),
        ]
    return (
        interactive_state,
        gr.update(
            choices=interactive_state["multi_mask"]["mask_names"], value=mask_dropdown
        ),
        select_frame,
        [[], []],
        operation_log,
    )


def clear_click(video_state, click_state):
    click_state = [[], []]
    template_frame = video_state["origin_images"][video_state["select_frame_number"]]
    operation_log = [
        ("", ""),
        ("Clear points history and refresh the image.", "Normal"),
    ]
    return template_frame, click_state, operation_log


def remove_multi_mask(interactive_state, mask_dropdown):
    interactive_state["multi_mask"]["mask_names"] = []
    interactive_state["multi_mask"]["masks"] = []

    operation_log = [("", ""), ("Remove all mask, please add new masks", "Normal")]
    return interactive_state, gr.update(choices=[], value=[]), operation_log


def show_mask(video_state, interactive_state, mask_dropdown):
    mask_dropdown.sort()
    select_frame = video_state["origin_images"][video_state["select_frame_number"]]
    for i in range(len(mask_dropdown)):
        mask_number = int(mask_dropdown[i].split("_")[1]) - 1
        mask = interactive_state["multi_mask"]["masks"][mask_number]
        select_frame = mask_painter(
            select_frame, mask.astype("uint8"), mask_color=mask_number + 2
        )

    operation_log = [
        ("", ""),
        ("Select {} for tracking or inpainting".format(mask_dropdown), "Normal"),
    ]
    return select_frame, operation_log
