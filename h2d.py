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
    prompt_with_breed = f"dogs face,{breed},dog,face,Portrait"
    # Step 3: ControlNet 유닛 생성
    unit1 = webuiapi.ControlNetUnit(input_image=img, module='mediapipe_face',pixel_perfect=True,control_mode=2)
    unit2 = webuiapi.ControlNetUnit(input_image=img, module='reference_only')
    unit3 = webuiapi.ControlNetUnit(input_image=img, module='depth_leres++',pixel_perfect=True,control_mode=2)
    r2 = api.img2img(prompt=prompt_with_breed,
                     negative_prompt="painting, human, low quality,clothes,body,people,man,woman",
                     images=[img],
                     width=512,
                     height=512,
                     controlnet_units=[unit1,unit2,unit3],
                     sampler_name="DPM++ 2M SDE Karras",
                     steps=40,
                     cfg_scale=6.5,
                     denoising_strength=0.85,
                     override_settings={
                         "sd_model_checkpoint":"v1-5-pruned-emaonly-h2d.ckpt"},
                     override_settings_restore_afterwards=True,
                     )

    return breed, r2.image

iface = gr.Interface(
    fn=call_apis_with_controlnet,
    inputs=gr.Image(type="pil", label="이미지를 넣어주세요"),
    outputs=[gr.Textbox(label="닮은 견종은?"), gr.Image(type="pil", label="You like...")],  # Include a textbox for breed
)

iface.launch()