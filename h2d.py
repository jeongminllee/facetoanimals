import webuiapi
import gradio as gr
from dogbreedfind import predict_breed

# API 클라이언트 생성 및 인증
api = webuiapi.WebUIApi()
api.controlnet_model_list()

def call_apis_with_controlnet(input_image):
    # Step 1: 이미지
    img = input_image
    # Step 2: 이미지를 품종분류함수에
    breed = predict_breed(img)
    #분류된 품종이 포함된 이미지 생성
    prompt_with_breed = f"dogs face, {breed}"
    # Step 3: ControlNet 유닛 생성
    unit1 = webuiapi.ControlNetUnit(input_image=img, module='mediapipe_face')
    unit2 = webuiapi.ControlNetUnit(input_image=img, module='reference_only')
    r2 = api.img2img(prompt=prompt_with_breed,
                     negative_prompt="paint, human",
                     images=[img],
                     width=512,
                     height=512,
                     controlnet_units=[unit1,unit2],
                     sampler_name="DPM++ 2M SDE Heun Karras",
                     cfg_scale=6,
                     denoising_strength=0.8,
                     override_settings={
                         "sd_model_checkpoint":"v1-5-pruned-emaonly-h2d.ckpt"},
                     override_settings_restore_afterwards=True,
                     )

    return breed, r2.image


custom_css = """
    .input_interface, .output_interface {
        width: 45%;
    }
    .interface {
        justify-content: space-between;
    }
"""


iface = gr.Interface(
    fn=call_apis_with_controlnet,
    inputs=gr.Image(type="pil", label="이미지를 넣어주세요"),
    outputs=[gr.Textbox(label="닮은 견종은?"), gr.Image(type="pil", label="You like...")],  # Include a textbox for breed
)

iface.launch()