tts --text "$1" \
 --model_path encoder_path/best_model.pth \
 --config_path encoder_path/config.json \
 --vocoder_path best_model.pth \
 --vocoder_config_path config.json \
 --out_path test.wav && \
 aplay test.wav