from ko_pron import romanise
from espeakng import ESpeakNG
from phonemizer import phonemize

print(romanise("있다", "ipa"))

esng = ESpeakNG()

for v in esng.voices:
    if v["language"] == "ko":
        print(v["language"],v["voice_name"])
esng.voice = "korean"
#esng.say("안녕하세요?")
ipa = esng.g2p('안녕하세요?', ipa=2)
print(ipa)

p = phonemize('안녕하세요', language='ko')
print(p)