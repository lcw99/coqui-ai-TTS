tts --text "$1" \
 --model_path ../../glow_tts/colab-run/best_model.pth \
 --config_path ../../glow_tts/colab-run/config.json \
 --vocoder_path best_model.pth \
 --vocoder_config_path config.json \
 --out_path test_colab.wav && \
 aplay test_colab.wav