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
    """画像の向き補正とリサイズ（768px上限で動画生成に最適化）"""
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")

    max_dim = 768
    if max(img.width, img.height) > max_dim:
        scale = max_dim / max(img.width, img.height)
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

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

def call_hf_video_api(image_path: str, prompt: str) -> str:
    """Hugging Face Space APIを安全に呼び出す"""
    try:
        # 1. 相手のSpace定義を確認
        client = Client("multimodalart/wan2-1-fast", token=hf_token)
        api_info = client.view_api(all_endpoints=True, return_format="dict")
        endpoints = api_info.get("named_endpoints", {})
        
        # エンドポイントの引数構成をログに出力
        target_ep = "/generate_video" if "/generate_video" in endpoints else list(endpoints.keys())[0]
        params = endpoints[target_ep].get("parameters", [])
        print(f"[Endpoint Selected]: {target_ep}, required params count: {len(params)}")

        # パラメータ数に応じて適切な引数を詰める
        args = [handle_file(image_path), prompt]
        if len(params) >= 3:
            # ネガティブプロンプト
            args.append("camera movement, blurry, jerky, low quality")
        # 4つ目以降にシード値やステップ数がある場合のダミー引数補完
        for param in params[len(args):]:
            default_val = param.get("parameter_default")
            if default_val is not None:
                args.append(default_val)
            else:
                args.append(0)

        job = client.submit(*args, api_name=target_ep)
        result = job.result()
        return result

    except Exception as e1:
        print(f"[Wan2.1 Error]: {e1}, trying LTX-Video fallback...")
        # Wan2.1が過負荷または引数不一致の場合、軽量なLTX-Videoにフォールバック
        client_fallback = Client("Lightricks/LTX-Video", token=hf_token)
        job = client_fallback.submit(
            handle_file(image_path),
            prompt,
            "camera movement, low quality",
            api_name="/generate_video"
        )
        return job.result()

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
        result = call_hf_video_api(safe_image_path, motion_prompt)
    except Exception as e:
        print(f"[Generation Final Error]: {repr(e)}")
        raise gr.Error(f"動画生成に失敗しました: {e}")
    finally:
        if os.path.exists(safe_image_path):
            os.remove(safe_image_path)

    if isinstance(result, (list, tuple)):
        result = result[0]

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
