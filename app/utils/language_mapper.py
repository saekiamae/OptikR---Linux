"""
Language Code Mapper Utility

Converts between different OCR engine language code formats:
- ISO 639-1 (2-letter codes): en, de, es, fr, ja, etc.
- Tesseract (3-letter codes): eng, deu, spa, fra, jpn, etc.
- EasyOCR (ISO 639-1 + special codes): en, de, ch_sim, ch_tra, etc.
- Windows OCR / winocr (BCP-47 tags): en, ja, zh-Hans, sr-Latn, etc.
- RapidOCR (ISO 639-1 + zh_sim/zh_tra): en, ja, zh_sim, etc.
- DocTR (ISO 639-1): en, fr, de, etc.
- Surya OCR (ISO 639-1): en, ja, bn, ta, etc.
"""

import logging


class LanguageCodeMapper:
    """
    Maps between different OCR engine language code formats.
    
    Supports conversion between:
    - ISO 639-1 (2-letter): Used by EasyOCR, DocTR, Surya OCR, ONNX, Translation engines
    - Tesseract (3-letter): Used by Tesseract OCR
    - BCP-47 tags: Used by Windows OCR (winocr)
    - PaddleOCR codes: Used by PaddleOCR and RapidOCR
    """
    
    # ISO 639-1 (2-letter) to Tesseract (3-letter) mapping
    ISO_TO_TESSERACT: dict[str, str] = {
        'en': 'eng',      # English
        'de': 'deu',      # German
        'es': 'spa',      # Spanish
        'fr': 'fra',      # French
        'ja': 'jpn',      # Japanese
        'zh': 'chi_sim',  # Chinese (Simplified)
        'ko': 'kor',      # Korean
        'ru': 'rus',      # Russian
        'it': 'ita',      # Italian
        'pt': 'por',      # Portuguese
        'nl': 'nld',      # Dutch
        'pl': 'pol',      # Polish
        'tr': 'tur',      # Turkish
        'ar': 'ara',      # Arabic
        'hi': 'hin',      # Hindi
        'th': 'tha',      # Thai
        'vi': 'vie',      # Vietnamese
        'id': 'ind',      # Indonesian
        'uk': 'ukr',      # Ukrainian
        'cs': 'ces',      # Czech
        'sv': 'swe',      # Swedish
        'da': 'dan',      # Danish
        'fi': 'fin',      # Finnish
        'no': 'nor',      # Norwegian
        'hu': 'hun',      # Hungarian
        'ro': 'ron',      # Romanian
        'bg': 'bul',      # Bulgarian
        'el': 'ell',      # Greek
        'he': 'heb',      # Hebrew
        'fa': 'fas',      # Persian
        'hr': 'hrv',      # Croatian
        'sk': 'slk',      # Slovak
        'sl': 'slv',      # Slovenian
        'sr': 'srp',      # Serbian
        'bn': 'ben',      # Bengali
        'ta': 'tam',      # Tamil
        'te': 'tel',      # Telugu
        'ml': 'mal',      # Malayalam
        'mr': 'mar',      # Marathi
        'gu': 'guj',      # Gujarati
        'kn': 'kan',      # Kannada
        'pa': 'pan',      # Punjabi
        'ur': 'urd',      # Urdu
        'ms': 'msa',      # Malay
        'my': 'mya',      # Burmese
        'km': 'khm',      # Khmer
        'lo': 'lao',      # Lao
        'ka': 'kat',      # Georgian
    }
    
    # Tesseract to ISO 639-1 (reverse mapping)
    TESSERACT_TO_ISO: dict[str, str] = {v: k for k, v in ISO_TO_TESSERACT.items()}
    
    # Alternative Tesseract codes (some languages have multiple codes)
    TESSERACT_ALTERNATIVES: dict[str, str] = {
        'ger': 'deu',      # German alternative
        'fre': 'fra',      # French alternative
        'dut': 'nld',      # Dutch alternative
        'chi_tra': 'chi_tra',  # Chinese Traditional (keep as-is)
        'chi_sim': 'chi_sim',  # Chinese Simplified (keep as-is)
    }
    
    # EasyOCR special codes (codes that don't follow ISO 639-1)
    EASYOCR_SPECIAL: dict[str, str] = {
        'ch_sim': 'chi_sim',   # Chinese Simplified
        'ch_tra': 'chi_tra',   # Chinese Traditional
        'rs_cyrillic': 'rs_cyrillic',  # Serbian Cyrillic
    }
    
    # ISO 639-1 to PaddleOCR language code mapping
    ISO_TO_PADDLEOCR: dict[str, str] = {
        'en': 'en',
        'ja': 'japan',
        'ko': 'korean',
        'zh': 'ch',
        'de': 'german',
        'fr': 'french',
        'es': 'spanish',
        'it': 'it',
        'pt': 'pt',
        'ru': 'ru',
        'ar': 'ar',
        'hi': 'hi',
        'tr': 'tr',
    }
    
    # ISO 639-1 to Windows OCR BCP-47 tag mapping.
    # Most ISO codes pass through unchanged; only exceptions are listed.
    # Use .get(code, code) for default pass-through behaviour.
    ISO_TO_WINOCR: dict[str, str] = {
        'zh': 'zh-Hans',    # Simplified Chinese
        'zh_sim': 'zh-Hans',
        'zh_tra': 'zh-Hant',
        'no': 'nb',         # Norwegian -> Bokmål (Windows default)
        'sr': 'sr-Latn',    # Serbian Latin script
    }

    # ISO 639-1 to RapidOCR language code mapping.
    # RapidOCR (PaddleOCR via ONNX) uses ISO codes for most languages,
    # but distinguishes Chinese Simplified / Traditional explicitly.
    ISO_TO_RAPIDOCR: dict[str, str] = {
        'en': 'en',
        'ja': 'ja',
        'ko': 'ko',
        'zh': 'zh_sim',     # Default Chinese -> Simplified
        'de': 'de',
        'fr': 'fr',
        'es': 'es',
        'ru': 'ru',
        'ar': 'ar',
        'it': 'it',
        'pt': 'pt',
        'nl': 'nl',
        'tr': 'tr',
        'pl': 'pl',
        'vi': 'vi',
    }

    # Supported language sets for engines that use plain ISO 639-1 codes.
    DOCTR_SUPPORTED: list[str] = [
        'en', 'fr', 'de', 'es', 'pt', 'it', 'nl', 'pl', 'ru', 'ar',
        'zh', 'ja', 'ko', 'vi', 'hi', 'cs', 'ro', 'hu', 'sv', 'da',
    ]

    SURYA_SUPPORTED: list[str] = [
        'en', 'ja', 'ko', 'zh', 'de', 'fr', 'es', 'ru', 'ar', 'it', 'pt', 'nl',
        'tr', 'pl', 'hi', 'th', 'vi', 'id', 'cs', 'ro', 'hu', 'sv', 'da', 'fi',
        'el', 'uk', 'he', 'bn', 'ta', 'te', 'ml', 'mr', 'gu', 'kn', 'pa', 'ur',
        'fa', 'ms', 'my', 'km', 'lo', 'ka',
    ]

    WINOCR_SUPPORTED: list[str] = [
        'en', 'ar', 'bg', 'cs', 'da', 'de', 'el', 'es', 'fi', 'fr',
        'he', 'hi', 'hr', 'hu', 'id', 'it', 'ja', 'ko', 'nl', 'no',
        'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'sr', 'sv', 'th', 'tr',
        'uk', 'vi', 'zh',
    ]

    # Language name to ISO code mapping (for UI)
    NAME_TO_ISO: dict[str, str] = {
        'English': 'en',
        'German': 'de',
        'Spanish': 'es',
        'French': 'fr',
        'Japanese': 'ja',
        'Chinese': 'zh',
        'Chinese (Simplified)': 'zh',
        'Chinese (Traditional)': 'zh',
        'Korean': 'ko',
        'Russian': 'ru',
        'Italian': 'it',
        'Portuguese': 'pt',
        'Dutch': 'nl',
        'Polish': 'pl',
        'Turkish': 'tr',
        'Arabic': 'ar',
        'Hindi': 'hi',
        'Thai': 'th',
        'Vietnamese': 'vi',
        'Indonesian': 'id',
        'Ukrainian': 'uk',
        'Czech': 'cs',
        'Swedish': 'sv',
        'Danish': 'da',
        'Finnish': 'fi',
        'Norwegian': 'no',
        'Hungarian': 'hu',
        'Romanian': 'ro',
        'Bulgarian': 'bg',
        'Greek': 'el',
        'Hebrew': 'he',
        'Persian': 'fa',
        'Croatian': 'hr',
        'Slovak': 'sk',
        'Slovenian': 'sl',
        'Serbian': 'sr',
        'Bengali': 'bn',
        'Tamil': 'ta',
        'Telugu': 'te',
        'Malayalam': 'ml',
        'Marathi': 'mr',
        'Gujarati': 'gu',
        'Kannada': 'kn',
        'Punjabi': 'pa',
        'Urdu': 'ur',
        'Malay': 'ms',
        'Burmese': 'my',
        'Khmer': 'km',
        'Lao': 'lo',
        'Georgian': 'ka',
    }
    
    # ISO code to language name (reverse mapping)
    ISO_TO_NAME: dict[str, str] = {v: k for k, v in NAME_TO_ISO.items() if k in [
        'English', 'German', 'Spanish', 'French', 'Japanese', 'Chinese',
        'Korean', 'Russian', 'Italian', 'Portuguese', 'Dutch', 'Polish',
        'Turkish', 'Arabic', 'Hindi', 'Thai', 'Vietnamese', 'Indonesian',
        'Ukrainian', 'Czech', 'Swedish', 'Danish', 'Finnish', 'Norwegian',
        'Hungarian', 'Romanian', 'Bulgarian', 'Greek', 'Hebrew', 'Persian',
        'Croatian', 'Slovak', 'Slovenian', 'Serbian',
        'Bengali', 'Tamil', 'Telugu', 'Malayalam', 'Marathi', 'Gujarati',
        'Kannada', 'Punjabi', 'Urdu', 'Malay', 'Burmese', 'Khmer',
        'Lao', 'Georgian',
    ]}
    
    @classmethod
    def to_easyocr(cls, code: str) -> str:
        """
        Convert any language code to EasyOCR format (ISO 639-1).
        
        Args:
            code: Language code in any format
            
        Returns:
            Language code in EasyOCR format
        """
        if not code:
            return 'en'
        
        code = code.lower().strip()
        
        # Already in ISO format (2 letters)
        if len(code) == 2 and code in cls.ISO_TO_TESSERACT:
            return code
        
        # Check if it's a special EasyOCR code
        if code in cls.EASYOCR_SPECIAL:
            return code
        
        # Convert from Tesseract format
        if code in cls.TESSERACT_TO_ISO:
            return cls.TESSERACT_TO_ISO[code]
        
        # Check Tesseract alternatives
        if code in cls.TESSERACT_ALTERNATIVES:
            tesseract_code = cls.TESSERACT_ALTERNATIVES[code]
            if tesseract_code in cls.TESSERACT_TO_ISO:
                return cls.TESSERACT_TO_ISO[tesseract_code]
        
        # Try to extract first 2 letters if it's a longer code
        if len(code) > 2:
            potential_iso = code[:2]
            if potential_iso in cls.ISO_TO_TESSERACT:
                return potential_iso
        
        # Default to English
        logging.warning(f"Unknown language code '{code}', defaulting to 'en'")
        return 'en'
    
    @classmethod
    def to_tesseract(cls, code: str) -> str:
        """
        Convert any language code to Tesseract format.
        
        Args:
            code: Language code in any format
            
        Returns:
            Language code in Tesseract format
        """
        if not code:
            return 'eng'
        
        code = code.lower().strip()
        
        # Already in Tesseract format (3+ letters)
        if code in cls.TESSERACT_TO_ISO or code in cls.TESSERACT_ALTERNATIVES:
            return code
        
        # Convert from ISO format
        if code in cls.ISO_TO_TESSERACT:
            return cls.ISO_TO_TESSERACT[code]
        
        # Handle special cases
        if code == 'zh' or code == 'chi_sim' or code == 'ch_sim':
            return 'chi_sim'
        if code == 'chi_tra' or code == 'ch_tra':
            return 'chi_tra'
        
        # Default to English
        logging.warning(f"Unknown language code '{code}', defaulting to 'eng'")
        return 'eng'
    
    @classmethod
    def to_paddleocr(cls, code: str) -> str:
        """
        Convert any language code to PaddleOCR format.
        
        Args:
            code: Language code in any format
            
        Returns:
            Language code in PaddleOCR format
        """
        if not code:
            return 'en'
        
        code = code.lower().strip()
        
        # Already a known PaddleOCR code
        if code in cls.ISO_TO_PADDLEOCR.values():
            return code
        
        # Convert from ISO format
        if code in cls.ISO_TO_PADDLEOCR:
            return cls.ISO_TO_PADDLEOCR[code]
        
        # Convert from Tesseract format via ISO
        if code in cls.TESSERACT_TO_ISO:
            iso = cls.TESSERACT_TO_ISO[code]
            return cls.ISO_TO_PADDLEOCR.get(iso, 'en')
        
        # Default to English
        logging.warning(f"Unknown language code '{code}' for PaddleOCR, defaulting to 'en'")
        return 'en'
    
    @classmethod
    def to_winocr(cls, code: str) -> str:
        """
        Convert any language code to Windows OCR (BCP-47) format.
        
        Args:
            code: Language code in any format
            
        Returns:
            BCP-47 language tag for winocr
        """
        if not code:
            return 'en'

        iso = cls.to_easyocr(code)
        return cls.ISO_TO_WINOCR.get(iso, iso)

    @classmethod
    def to_rapidocr(cls, code: str) -> str:
        """
        Convert any language code to RapidOCR format.
        
        Args:
            code: Language code in any format
            
        Returns:
            Language code in RapidOCR format
        """
        if not code:
            return 'en'

        code_l = code.lower().strip()

        # Handle Chinese variants explicitly
        if code_l in ('zh_tra', 'ch_tra', 'chi_tra'):
            return 'zh_tra'

        iso = cls.to_easyocr(code)
        return cls.ISO_TO_RAPIDOCR.get(iso, iso)

    @classmethod
    def to_doctr(cls, code: str) -> str:
        """
        Convert any language code to DocTR format (ISO 639-1).
        
        Args:
            code: Language code in any format
            
        Returns:
            ISO 639-1 language code for DocTR
        """
        if not code:
            return 'en'

        iso = cls.to_easyocr(code)

        if iso not in cls.DOCTR_SUPPORTED:
            logging.warning(f"Language '{code}' (iso={iso}) not in DocTR supported set, defaulting to 'en'")
            return 'en'
        return iso

    @classmethod
    def to_surya(cls, code: str) -> str:
        """
        Convert any language code to Surya OCR format (ISO 639-1).
        
        Args:
            code: Language code in any format
            
        Returns:
            ISO 639-1 language code for Surya OCR
        """
        if not code:
            return 'en'

        iso = cls.to_easyocr(code)

        if iso not in cls.SURYA_SUPPORTED:
            logging.warning(f"Language '{code}' (iso={iso}) not in Surya supported set, defaulting to 'en'")
            return 'en'
        return iso

    @classmethod
    def normalize(cls, code: str, target_engine: str) -> str:
        """
        Normalize language code for a specific OCR engine.
        
        Args:
            code: Language code in any format
            target_engine: Target OCR engine name
            
        Returns:
            Normalized language code for the target engine
            
        Engine-specific formats:
            - EasyOCR: ISO 639-1 (en, de, ja, etc.)
            - Tesseract: 3-letter codes (eng, deu, jpn, etc.)
            - PaddleOCR: PaddleOCR names (japan, korean, ch, etc.)
            - Windows OCR: BCP-47 tags (en, ja, zh-Hans, sr-Latn, etc.)
            - RapidOCR: ISO 639-1 + zh_sim/zh_tra
            - DocTR: ISO 639-1 (en, fr, de, etc.)
            - Surya OCR: ISO 639-1 (en, ja, bn, etc.)
            - ONNX: ISO 639-1 (en, de, ja, etc.)
            - Manga OCR: Always 'ja' (Japanese only)
        """
        if not code or not target_engine:
            return 'en'
        
        engine = target_engine.lower().strip()
        
        # Mokuro only supports Japanese
        if engine == 'mokuro':
            if code.lower() in ['ja', 'jpn', 'japanese']:
                return 'ja'
            logging.warning(f"Mokuro only supports Japanese, converting '{code}' to 'ja'")
            return 'ja'
        
        # Tesseract uses 3-letter codes (eng, deu, jpn, etc.)
        elif engine == 'tesseract':
            return cls.to_tesseract(code)
        
        # PaddleOCR uses its own language names (japan, korean, ch, etc.)
        elif engine == 'paddleocr':
            return cls.to_paddleocr(code)
        
        # Windows OCR uses BCP-47 tags (en, ja, zh-Hans, sr-Latn, etc.)
        elif engine in ['windows_ocr', 'winocr']:
            return cls.to_winocr(code)

        # RapidOCR uses ISO 639-1 with zh_sim/zh_tra for Chinese
        elif engine == 'rapidocr':
            return cls.to_rapidocr(code)

        # DocTR uses ISO 639-1
        elif engine == 'doctr':
            return cls.to_doctr(code)

        # Surya OCR uses ISO 639-1
        elif engine in ['surya_ocr', 'surya']:
            return cls.to_surya(code)

        # EasyOCR and ONNX use ISO 639-1 format (en, de, ja, etc.)
        elif engine in ['easyocr', 'onnx']:
            return cls.to_easyocr(code)
        
        # Unknown engine, default to ISO format
        else:
            logging.warning(f"Unknown OCR engine '{target_engine}', using ISO format")
            return cls.to_easyocr(code)
    
    @classmethod
    def from_name(cls, name: str) -> str:
        """
        Convert language name to ISO 639-1 code.
        
        Args:
            name: Language name (e.g., 'English', 'German')
            
        Returns:
            ISO 639-1 language code
        """
        if not name:
            return 'en'
        
        name = name.strip()
        return cls.NAME_TO_ISO.get(name, 'en')
    
    @classmethod
    def to_name(cls, code: str) -> str:
        """
        Convert language code to language name.
        
        Args:
            code: Language code in any format
            
        Returns:
            Language name
        """
        if not code:
            return 'English'
        
        # First normalize to ISO format
        iso_code = cls.to_easyocr(code)
        
        # Get name from mapping
        return cls.ISO_TO_NAME.get(iso_code, 'English')
    
    @classmethod
    def is_valid_code(cls, code: str, engine: str | None = None) -> bool:
        """
        Check if a language code is valid.
        
        Args:
            code: Language code to validate
            engine: Optional engine name to validate against
            
        Returns:
            True if code is valid, False otherwise
        """
        if not code:
            return False
        
        code = code.lower().strip()
        
        # Check ISO format
        if code in cls.ISO_TO_TESSERACT:
            return True
        
        # Check Tesseract format
        if code in cls.TESSERACT_TO_ISO or code in cls.TESSERACT_ALTERNATIVES:
            return True
        
        # Check special codes
        if code in cls.EASYOCR_SPECIAL:
            return True
        
        return False
    
    @classmethod
    def get_supported_languages(cls, engine: str) -> list:
        """
        Get list of supported language codes for an engine.
        
        Args:
            engine: OCR engine name
            
        Returns:
            List of supported language codes
        """
        engine = engine.lower().strip()
        
        if engine in ['windows_ocr', 'winocr']:
            return list(cls.WINOCR_SUPPORTED)
        elif engine == 'rapidocr':
            return list(cls.ISO_TO_RAPIDOCR.keys())
        elif engine == 'doctr':
            return list(cls.DOCTR_SUPPORTED)
        elif engine in ['surya_ocr', 'surya']:
            return list(cls.SURYA_SUPPORTED)
        elif engine in ['easyocr', 'paddleocr', 'onnx']:
            return list(cls.ISO_TO_TESSERACT.keys())
        elif engine == 'tesseract':
            return list(cls.TESSERACT_TO_ISO.keys())
        else:
            return list(cls.ISO_TO_TESSERACT.keys())


# Convenience functions for quick access
def normalize_language_code(code: str, engine: str) -> str:
    """Convenience function to normalize language code."""
    return LanguageCodeMapper.normalize(code, engine)


def language_name_to_code(name: str) -> str:
    """Convenience function to convert language name to code."""
    return LanguageCodeMapper.from_name(name)


def language_code_to_name(code: str) -> str:
    """Convenience function to convert language code to name."""
    return LanguageCodeMapper.to_name(code)
