tts --text "$1" \
 --model_path ../../glow_tts/run-August-19-2022_10+44AM-cda630e0/best_model.pth \
 --config_path ../../glow_tts/run-August-19-2022_10+44AM-cda630e0/config.json \
 --vocoder_path best_model.pth \
 --vocoder_config_path config.json \
 --out_path test.wav && \
 aplay test.wav