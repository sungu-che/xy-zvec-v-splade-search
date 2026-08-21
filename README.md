

# Language Support Matrix (Language -> English Translation)

All supported languages are translated into **English (`eng`)**.

| Language Code | Language Name | Translation Model | Detection Method | Status |
| :---: | :--- | :--- | :--- | :---: |
| **kor** | Korean (한국어) | `opus-mt_tiny_kor-eng` | Unicode `\uac00-\ud7a3` | ✅ Fully Supported |
| **zho** | Chinese (中文) | `opus-mt_tiny_zho-eng` | Unicode `\u4e00-\u9fff` | ✅ Fully Supported |
| **jpn** | Japanese (日本語) | *(No model)* | - | ⏳ Planned for Future Support |
| **ara** | Arabic (العربية) | `opus-mt_tiny_ara-eng` | Unicode `\u0600-\u06ff` | ✅ Fully Supported |
| **rus** | Russian (Русский) | `opus-mt_tiny_rus-eng` | Unicode `\u0400-\u04ff` | ✅ Fully Supported |
| **ell** | Greek (Ελληνικά) | `opus-mt_tiny_ell-eng` | Unicode `\u0370-\u03ff` | ✅ Fully Supported |
| **fra** | French (Français) | `opus-mt_tiny_fra-eng` | Always Attempt | ✅ Fully Supported |
| **deu** | German (Deutsch) | `opus-mt_tiny_deu-eng` | Always Attempt | ✅ Fully Supported |
| **ita** | Italian (Italiano) | `opus-mt_tiny_ita-eng` | Always Attempt | ✅ Fully Supported |
| **spa** | Spanish (Español) | `opus-mt_tiny_spa-eng` | Always Attempt | ✅ Fully Supported |
| **nld** | Dutch (Nederlands) | `opus-mt_tiny_nld-eng` | Always Attempt | ✅ Fully Supported |
| **tur** | Turkish (Türkçe) | `opus-mt_tiny_tur-eng` | Always Attempt | ✅ Fully Supported |
| **cat** | Catalan (Català) | `opus-mt_tiny_cat-eng` | Always Attempt | ✅ Fully Supported |
| **eng** | English | N/A (Original) | Use Original | ✅ Fully Supported |

> **Note:**
> * All languages are set to be translated into **English**.
> * Japanese (**jpn**) is currently unsupported due to the lack of a translation model, but will be integrated in a future update when supported.

# License

## Reference License

This project integrates and extends the following open-source works:

| Project | Component | License |
|-----------|---------|---------|
| [Helsinki-NLP/opustranslate](https://huggingface.co/collections/Helsinki-NLP/opustranslate) | Translate | Apache-2.0 |
| [timm/ViT-SO400M-16-SigLIP2-256](https://huggingface.co/timm/ViT-SO400M-16-SigLIP2-256) | Zero-Shot Image Classification | Apache-2.0 |
| [KRAFTON/Raon-VisionEncoder](https://huggingface.co/KRAFTON/Raon-VisionEncoder) | Embedding Model | Apache-2.0 |
| [KRAFTON/Raon-VisionEncoder](https://huggingface.co/KRAFTON/Raon-VisionEncoder) | Embedding Engine | Apache-2.0 |
| [alibaba/zvec](https://github.com/alibaba/zvec) | Embedding Engine | Apache-2.0 |

