import os
from re import T

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig, CharactersConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from dataclasses import dataclass, field

output_path = os.path.dirname(os.path.abspath(__file__))
dataset_config = BaseDatasetConfig(
    name="kss_ko", meta_file_train="transcript.v.1.4.txt", language="ko-kr", path="/home/chang/bighard/AI/tts/dataset/kss/"
)

audio_config = VitsAudioConfig(
    sample_rate=22050, win_length=1024, hop_length=256, num_mels=80, mel_fmin=0, mel_fmax=None,
)

config = VitsConfig(
    audio=audio_config,
    run_name="vits_kss_ko_no_phoneme_tokenize",
    batch_size=12,
    eval_batch_size=12,
    batch_group_size=5,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="korean_phoneme_cleaners_with_tokeniner",
    use_phonemes=False,
    phoneme_language="ko",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache_g2p_ko"),
    compute_input_seq_cache=True,
    eval_split_size=10,
    print_step=50,
    save_step=1000,
    print_eval=True,
    mixed_precision=True,
    output_path=output_path,
    datasets=[dataset_config],
    cudnn_benchmark=False,
    min_audio_len=32 * 256 * 4,
    max_audio_len=220500,
    test_sentences = [
        ["목소리를 만드는데는 오랜 시간이 걸린다, 인내심이 필요하다."],
        ["목소리가 되어라, 메아리가 되지말고."],
        ["철수야 미안하다. 아무래도 그건 못하겠다."],
        ["이 케익은 정말 맛있다. 촉촉하고 달콤하다."],
        ["1963년 11월 23일 이전"],
    ],
    # test_sentences = [
    #         ["It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent."],
    #         ["Be a voice, not an echo."],
    #         ["I'm sorry Dave. I'm afraid I can't do that."],
    #         ["This cake is great. It's so delicious and moist."],
    #         ["Prior to November 22, 1963."],
    #     ]
    #use_language_weighted_sampler=True,
    characters=CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="<PAD>",
        eos="<EOS>",
        bos="<BOS>",
        blank="<BLNK>",
        characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄋᄌᄍᄎᄏᄐᄑ"+"ᄒ"+"ᅡᅢᅣᅤᅥᅦᅧᅨᅩᅪᅫᅬᅭᅮᅯᅰᅱᅲᅳᅴᅵᆨᆩᆪᆫᆬᆭᆮᆯᆰᆱᆲᆳᆴᆵᆶᆷᆸᆹᆺᆻᆼᆽᆾᆿᇀᇁᇂ",
        punctuations="!¡'(),-.:;¿? ",
        phonemes=None,
    ),
)

# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# config is updated with the default characters if not defined in the config.
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
    return items

# load training samples
train_samples, eval_samples = load_tts_samples(dataset_config, 
    eval_split=True, 
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
    formatter=formatter)

# init model
model = Vits(config, ap, tokenizer, speaker_manager=None)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()
