tts --text "소년은  개울가에서 소녀를 보자, 곧 윤초시네 증손녀 딸이라는 걸 알 수 있었다." \
 --model_path ../../glow_tts/run-August-19-2022_10+44AM-cda630e0/best_model.pth \
 --config_path ../../glow_tts/run-August-19-2022_10+44AM-cda630e0/config.json \
 --vocoder_path best_model.pth \
 --vocoder_config_path config.json \
 --out_path test.wav && \
 aplay test.wav