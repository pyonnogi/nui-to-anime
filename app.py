import os
import shutil
import tempfile
import uuid
from dotenv import load_dotenv
from google import genai
import gradio as gr
from gradio_client import Client, handle_file
from PIL import Image, ImageOps

load_dotenv()

# クライアント初期化
gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=gemini_api_key)
hf_token = os.getenv("HF_TOKEN")

def prepare_image(image_path: str) -> str:
    """EXIF向きを正し、512x512の正方形に中央クロップ＆リサイズ（Diffusion I2Vの標準入力形式）"""
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")

    # Diffusion系モデルが最もエラーを起こさない 512x512 にセンタリングクロップ
    target_size = 512
    width, height = img.size
    min_dim = min(width, height)
    
    left = (width - min_dim) / 2
    top = (height - min_dim) / 2
    right = (width + min_dim) / 2
    bottom = (height + min_dim) / 2
    img = img.crop((left, top, right, bottom))
    img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)

    temp_dir = tempfile.gettempdir()
    processed_path = os.path.join(temp_dir, f"input_{uuid.uuid4().hex[:8]}.jpg")
    img.save(processed_path, format="JPEG", quality=90)
    return processed_path

def generate_motion_prompt(image_path: str, instruction: str) -> str:
    """Gemini 2.5 Flash で動画指示文を最適化"""
    try:
        img = Image.open(image_path)
        prompt = f"""
        The user wants to animate this plushie/toy.
        User instruction: {instruction}
        
        Output ONLY a concise, high-quality prompt in English for an Image-to-Video diffusion model.
        Requirements:
        - Keep background static.
        - Subtle, natural, cute movement.
        - Output just the raw prompt text without extra explanation.
        """
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[img, prompt],
        )
        return response.text.strip()
    except Exception as e:
        print(f"[Gemini Error]: {e}")
        return f"A plushie subtly moving, cute motion, {instruction}, static background"

def generate_video_api(image_path: str, prompt: str) -> str:
    """
    安定稼働している無料動画生成Spaceを順次呼び出し
    """
    # 候補1: svd (Stable Video Diffusion - 公式標準)
    try:
        print("[Video Gen]: Calling multimodalart/stable-video-diffusion...")
        client = Client("multimodalart/stable-video-diffusion", token=hf_token)
        result = client.predict(
            handle_file(image_path), # input image
            0,                       # seed
            25,                      # motion bucket id (1-255: 小さめの動き)
            6,                       # fps
            api_name="/video"
        )
        if result:
            return result
    except Exception as e:
        print(f"[SVD Error]: {e}")

    # 候補2: ali-vilab/i2vgen-xl (Alibaba Image-to-Video)
    try:
        print("[Video Gen]: Calling ali-vilab/i2vgen-xl...")
        client = Client("ali-vilab/i2vgen-xl", token=hf_token)
        result = client.predict(
            handle_file(image_path),
            prompt,
            api_name="/run"
        )
        if result:
            return result
    except Exception as e:
        print(f"[I2VGen-XL Error]: {e}")

    raise RuntimeError("現在無料の動画生成サーバーが混雑または停止しています。少し時間を置いて再試行してください。")

def process_animation(image_data, instruction: str):
    if image_data is None:
        raise gr.Error("画像をアップロードしてください。")
    if not instruction:
        instruction = "Subtle wiggle and breathing movement"

    raw_path = image_data["background"] if isinstance(image_data, dict) else image_data
    safe_image_path = prepare_image(raw_path)

    # 1. Geminiでプロンプト生成
    motion_prompt = generate_motion_prompt(safe_image_path, instruction)
    print(f"[Generated Prompt]: {motion_prompt}")

    # 2. 動画生成実行
    try:
        result = generate_video_api(safe_image_path, motion_prompt)
    except Exception as e:
        print(f"[Generation Final Error]: {repr(e)}")
        raise gr.Error(str(e))
    finally:
        if os.path.exists(safe_image_path):
            os.remove(safe_image_path)

    if isinstance(result, (list, tuple)):
        result = result[0]

    # 一時領域に退避
    temp_dir = tempfile.gettempdir()
    temp_output_path = os.path.join(temp_dir, f"temp_{os.path.basename(result)}")
    shutil.copy(result, temp_output_path)

    return temp_output_path, motion_prompt

# UI画面の構築
with gr.Blocks(title="nui-to-anime") as demo:
    gr.Markdown("# 🧸 nui-to-anime（ぬいぐるみアニメーション変換ツール）")
    gr.Markdown("写真をアップロードして指示を入力すると、動画をその場で生成して返します。")

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="filepath", label="写真アップロード", sources=["upload"])
            instruction = gr.Textbox(
                label="動きの指示",
                placeholder="例: 手を小さく振る、息をするように揺れる",
                value="手を小さく振る"
            )
            submit_btn = gr.Button("アニメーション生成", variant="primary")
        with gr.Column():
            output_vid = gr.Video(label="生成結果", autoplay=True, loop=True)
            output_prompt = gr.Textbox(label="最適化プロンプト", interactive=False)

    submit_btn.click(
        fn=process_animation,
        inputs=[input_img, instruction],
        outputs=[output_vid, output_prompt],
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=port
    )
