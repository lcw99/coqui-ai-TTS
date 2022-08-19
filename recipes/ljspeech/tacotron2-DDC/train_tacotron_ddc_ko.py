import os

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

# from TTS.tts.datasets.tokenizer import Tokenizer

output_path = os.path.dirname(os.path.abspath(__file__))

# init configs
dataset_config = BaseDatasetConfig(
    name="kss_ko",
    meta_file_train="transcript.v.1.4.txt",
    language="ko-kr",
    path="/home/chang/bighard/AI/tts/dataset/kss/",
)

audio_config = BaseAudioConfig(
    sample_rate=22050,
    resample=True,
    do_trim_silence=True,
    trim_db=60.0,
    signal_norm=False,
    mel_fmin=0.0,
    mel_fmax=8000,
    spec_gain=1.0,
    log_func="np.log",
    ref_level_db=20,
    preemphasis=0.0,
)

config = Tacotron2Config(  # This is the config that is saved for the future use
    audio=audio_config,
    batch_size=64,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    r=6,
    gradual_training=[[0, 6, 64], [10000, 4, 32], [50000, 3, 32], [100000, 2, 32]],
    double_decoder_consistency=True,
    epochs=1000,
    text_cleaner="phoneme_cleaners",
    use_phonemes=True,
    phoneme_language="ko",
    phonemizer="espeak",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache_ko"),
    precompute_num_workers=8,
    print_step=25,
    print_eval=True,
    mixed_precision=False,
    output_path=output_path,
    datasets=[dataset_config],
)

# init audio processor
ap = AudioProcessor(**config.audio.to_dict())

# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# If characters are not defined in the config, default characters are passed to the config
tokenizer, config = TTSTokenizer.init_from_config(config)

# LOAD DATA SAMPLES
# Each sample is a list of ```[text, audio_file_path, speaker_name]```
# You can define your custom sample loader returning the list of samples.
# Or define your custom formatter and pass it to the `load_tts_samples`.
# Check `TTS.tts.datasets.load_tts_samples` for more details.
# train_samples, eval_samples = load_tts_samples(
#     dataset_config,
#     eval_split=True,
#     eval_split_max_size=config.eval_split_max_size,
#     eval_split_size=config.eval_split_size,
# )
def formatter(root_path, manifest_file, **kwargs):  # pylint: disable=unused-argument
    """Assumes each line as ```<filename>|<transcription>```
    """
    txt_file = os.path.join(root_path, manifest_file)
    items = []
    speaker_name = "KBSVoice"
    with open(txt_file, "r", encoding="utf-8") as ttf:
        cnt = 0
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, cols[0])
            text = cols[1]
            items.append({"text":text, "audio_file":wav_file, "speaker_name":speaker_name})
            cnt += 1
            #if cnt >= 10000:
            #if cnt >= 1000:
            #    break
    return items

train_samples, eval_samples = load_tts_samples(
    dataset_config, 
    eval_split=True, 
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
    formatter=formatter)

# INITIALIZE THE MODEL
# Models take a config object and a speaker manager as input
# Config defines the details of the model like the number of layers, the size of the embedding, etc.
# Speaker manager is used by multi-speaker models.
model = Tacotron2(config, ap, tokenizer, speaker_manager=None)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)
trainer.fit()
