"""Record a VoiceGuard demo video with Playwright against the live domain."""

import pathlib
import re

from playwright.sync_api import sync_playwright

URL = "https://voice-deepfake-vishing-detector-generator.eu.cc/"
OUT = "/tmp/demo_raw"
pathlib.Path(OUT).mkdir(parents=True, exist_ok=True)


def run():
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True, args=["--no-sandbox"])
        ctx = b.new_context(
            viewport={"width": 1280, "height": 720},
            record_video_dir=OUT,
            record_video_size={"width": 1280, "height": 720},
        )
        pg = ctx.new_page()
        pause = pg.wait_for_timeout

        def click_text(rx, timeout=8000):
            pg.get_by_role("button", name=re.compile(rx, re.I)).first.click(timeout=timeout)

        pg.goto(URL, wait_until="networkidle", timeout=90000)
        pause(2500)

        # --- login ---
        try:
            click_text(r"log ?in")
            pause(1200)
            pg.locator("input[type=text]").last.fill("admin")
            pause(600)
            pg.locator("input[type=password]").last.fill("voiceguard2026")
            pause(600)
            pg.locator("input[type=password]").last.press("Enter")
            pause(3000)
        except Exception as e:
            print("login step:", e)

        # --- detect a PREMIUM ElevenLabs fake ---
        try:
            click_text(r"^\s*detect")
        except Exception:
            pass
        pause(1000)
        pg.locator("input[type=file]").set_input_files("/tmp/demo_clips/elevenlabs_fake.wav")
        pause(1800)
        try:
            click_text(r"detect deepfake")
            # wait until inference finishes (button reverts from "Analysing…")
            pg.get_by_role("button", name=re.compile("detect deepfake", re.I)).wait_for(
                state="visible", timeout=40000
            )
        except Exception as e:
            print("detect step:", e)
        pause(5000)  # hold on the FAKE verdict

        # --- generate watermarked speech ---
        try:
            click_text(r"^\s*generate")
            pause(1500)
            pg.locator("textarea").first.fill(
                "This is VoiceGuard generating watermarked synthetic speech for the demo."
            )
            pause(1500)
            click_text(r"synthes")
            # wait until synthesis finishes (button reverts from "Generating…")
            pg.get_by_role("button", name=re.compile("synthes", re.I)).wait_for(
                state="visible", timeout=40000
            )
            pause(4000)
        except Exception as e:
            print("generate step:", e)

        # --- results ---
        try:
            click_text(r"^\s*results")
            pause(5000)
        except Exception as e:
            print("results step:", e)

        ctx.close()
        b.close()
    vids = list(pathlib.Path(OUT).glob("*.webm"))
    print("VIDEOS:", [str(v) for v in vids])


if __name__ == "__main__":
    run()
