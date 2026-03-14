"""
Built-in context profile presets for the Context Manager plugin.

Provides factory functions that return pre-configured ContextProfile
instances for common use cases (manga, visual novels, etc.).
"""

from .context_profile import (
    ContextProfile,
    LockedTerm,
    TranslationMemoryEntry,
    RegexRule,
    FormattingRules,
)


def create_manga_preset() -> ContextProfile:
    """Create a preset profile for manga translation (JA → EN)."""
    profile = ContextProfile()
    profile.name = "Sample - Manga"
    profile.category = "Manga / Anime"
    profile.description = (
        "Generic manga and anime terminology: honorifics, cultural terms, "
        "common sound effects, and expressions. Japanese → English."
    )
    profile.source_language = "ja"
    profile.target_language = "en"

    ctx = profile.global_context

    # -- Honorifics (highest priority — must not be translated) -----------
    honorifics = [
        ("さん", "-san", "Polite; Mr./Ms."),
        ("くん", "-kun", "Familiar masculine"),
        ("ちゃん", "-chan", "Endearing / cute"),
        ("様", "-sama", "Very formal / respectful"),
        ("殿", "-dono", "Formal; lord/lady"),
        ("先輩", "-senpai", "Senior / upperclassman"),
        ("先生", "-sensei", "Teacher / master / doctor"),
        ("氏", "-shi", "Formal written honorific"),
    ]
    for src, tgt, note in honorifics:
        ctx.locked_terms.append(LockedTerm(
            source=src, target=tgt, type="honorific",
            case_sensitive=True, priority=200, notes=note,
        ))

    # -- Cultural terms that are commonly kept in romaji ----------------
    cultural = [
        ("忍者", "ninja", "general"),
        ("侍", "samurai", "general"),
        ("刀", "katana", "general"),
        ("道場", "dojo", "location"),
        ("弁当", "bento", "general"),
        ("着物", "kimono", "general"),
        ("布団", "futon", "general"),
        ("畳", "tatami", "general"),
        ("下駄", "geta", "general"),
        ("おにぎり", "onigiri", "general"),
        ("ラーメン", "ramen", "general"),
        ("味噌", "miso", "general"),
        ("寿司", "sushi", "general"),
        ("天ぷら", "tempura", "general"),
        ("居酒屋", "izakaya", "location"),
        ("温泉", "onsen", "location"),
        ("神社", "shrine", "location"),
        ("鳥居", "torii", "location"),
    ]
    for src, tgt, ttype in cultural:
        ctx.locked_terms.append(LockedTerm(
            source=src, target=tgt, type=ttype,
            case_sensitive=True, priority=150,
        ))

    # -- Family / relationship terms ------------------------------------
    family = [
        ("お兄ちゃん", "onii-chan", "Older brother (casual)"),
        ("お姉ちゃん", "onee-chan", "Older sister (casual)"),
        ("お兄さん", "onii-san", "Older brother (polite)"),
        ("お姉さん", "onee-san", "Older sister (polite)"),
        ("おじいちゃん", "ojii-chan", "Grandfather (casual)"),
        ("おばあちゃん", "obaa-chan", "Grandmother (casual)"),
        ("仲間", "nakama", "Comrades / close companions"),
    ]
    for src, tgt, note in family:
        ctx.locked_terms.append(LockedTerm(
            source=src, target=tgt, type="character",
            case_sensitive=True, priority=160, notes=note,
        ))

    # -- Common expressions / exclamations ------------------------------
    expressions = [
        ("いただきます", "itadakimasu", "Said before eating"),
        ("ごちそうさま", "gochisousama", "Said after eating"),
        ("お疲れ様", "otsukaresama", "Good work / thanks for your effort"),
        ("よろしくお願いします", "yoroshiku onegaishimasu", "Pleased to meet you / please take care of it"),
    ]
    for src, tgt, note in expressions:
        ctx.locked_terms.append(LockedTerm(
            source=src, target=tgt, type="general",
            case_sensitive=True, priority=140, notes=note,
        ))

    # -- Sound effects (SFX) — translation memory (softer match) -------
    sfx_entries = [
        ("ドキドキ", "*ba-dump ba-dump*", "Heartbeat / nervousness"),
        ("ゴゴゴ", "*menacing*", "Ominous rumbling (JoJo-style)"),
        ("バキ", "*crack*", "Breaking / impact"),
        ("ドン", "*boom*", "Explosion / dramatic impact"),
        ("ガタッ", "*clatter*", "Sudden movement / chair scraping"),
        ("シーン", "*silence*", "Dead silence"),
        ("ザワザワ", "*murmur murmur*", "Crowd chatter / unease"),
        ("キラキラ", "*sparkle*", "Sparkling / glittering"),
        ("ニコニコ", "*grin*", "Smiling"),
        ("ガーン", "*shock*", "Comedic shock"),
        ("ワクワク", "*excited*", "Excitement / anticipation"),
        ("ビクッ", "*flinch*", "Startled jump"),
    ]
    for src, tgt, note in sfx_entries:
        ctx.translation_memory.append(TranslationMemoryEntry(
            source=src, target=tgt, priority=80, notes=f"SFX: {note}",
        ))

    # -- Regex rules ----------------------------------------------------
    ctx.regex_rules = [
        RegexRule(
            pattern=r"「[^」]*」",
            action="preserve",
            stage="pre",
            description="Preserve text in Japanese quotation brackets 「」",
        ),
        RegexRule(
            pattern=r"『[^』]*』",
            action="preserve",
            stage="pre",
            description="Preserve text in double quotation brackets 『』",
        ),
        RegexRule(
            pattern=r"【[^】]*】",
            action="preserve",
            stage="pre",
            description="Preserve text in bold brackets 【】 (titles, labels)",
        ),
    ]

    # -- Formatting rules -----------------------------------------------
    ctx.formatting_rules = FormattingRules(
        preserve_honorifics=True,
        attack_uppercase=True,
        translate_sound_effects=True,
        preserve_brackets=True,
        language_style="casual",
    )

    return profile


def create_visual_novel_preset() -> ContextProfile:
    """Create a preset profile for visual novel translation (JA → EN)."""
    profile = ContextProfile()
    profile.name = "Sample - Visual Novel"
    profile.category = "Visual Novel"
    profile.description = (
        "Common visual novel UI terms, menu labels, and game system "
        "vocabulary. Japanese → English."
    )
    profile.source_language = "ja"
    profile.target_language = "en"

    ctx = profile.global_context

    # -- Menu / system UI labels ----------------------------------------
    ui_terms = [
        ("セーブ", "Save", "general", "Save game"),
        ("ロード", "Load", "general", "Load game"),
        ("オートモード", "Auto Mode", "general", "Auto-advance text"),
        ("スキップ", "Skip", "general", "Skip read text"),
        ("バックログ", "Backlog", "general", "Message history"),
        ("コンフィグ", "Config", "general", "Settings / configuration"),
        ("タイトル", "Title", "general", "Return to title screen"),
        ("クイックセーブ", "Quick Save", "general", ""),
        ("クイックロード", "Quick Load", "general", ""),
        ("既読スキップ", "Skip Read", "general", "Skip previously read text"),
        ("未読スキップ", "Skip Unread", "general", "Skip unread text"),
        ("環境設定", "Settings", "general", "Environment settings"),
        ("音量", "Volume", "general", "Audio volume"),
        ("ウィンドウ", "Window", "general", "Text window"),
        ("フルスクリーン", "Fullscreen", "general", ""),
        ("テキスト速度", "Text Speed", "general", ""),
        ("ギャラリー", "Gallery", "general", "CG / image gallery"),
        ("おまけ", "Extras", "general", "Bonus content"),
        ("シーン回想", "Scene Replay", "general", ""),
        ("次へ", "Next", "general", "Continue"),
        ("戻る", "Back", "general", "Go back"),
        ("はい", "Yes", "general", "Confirmation"),
        ("いいえ", "No", "general", "Rejection"),
        ("終了", "Quit", "general", "Exit game"),
    ]
    for src, tgt, ttype, note in ui_terms:
        ctx.locked_terms.append(LockedTerm(
            source=src, target=tgt, type=ttype,
            case_sensitive=True, priority=180, notes=note,
        ))

    # -- Honorifics (same core set as manga) ----------------------------
    honorifics = [
        ("さん", "-san"),
        ("くん", "-kun"),
        ("ちゃん", "-chan"),
        ("様", "-sama"),
        ("先輩", "-senpai"),
        ("先生", "-sensei"),
    ]
    for src, tgt in honorifics:
        ctx.locked_terms.append(LockedTerm(
            source=src, target=tgt, type="honorific",
            case_sensitive=True, priority=200,
        ))

    # -- Narrative / genre terms ----------------------------------------
    narrative = [
        ("選択肢", "choice", "Branching choice"),
        ("ルート", "route", "Story route / path"),
        ("エンディング", "ending", ""),
        ("トゥルーエンド", "true end", ""),
        ("バッドエンド", "bad end", ""),
        ("グッドエンド", "good end", ""),
        ("ノーマルエンド", "normal end", ""),
        ("攻略", "walkthrough", "Route completion guide"),
        ("立ち絵", "sprite", "Character standing image"),
        ("背景", "background", "Background art"),
    ]
    for src, tgt, note in narrative:
        ctx.locked_terms.append(LockedTerm(
            source=src, target=tgt, type="general",
            case_sensitive=True, priority=150, notes=note,
        ))

    # -- Common VN expressions (translation memory) --------------------
    ctx.translation_memory = [
        TranslationMemoryEntry(
            source="お兄ちゃん", target="onii-chan", priority=70,
            notes="Older brother (casual)",
        ),
        TranslationMemoryEntry(
            source="お姉ちゃん", target="onee-chan", priority=70,
            notes="Older sister (casual)",
        ),
        TranslationMemoryEntry(
            source="幼馴染", target="childhood friend", priority=60,
        ),
        TranslationMemoryEntry(
            source="転校生", target="transfer student", priority=60,
        ),
        TranslationMemoryEntry(
            source="生徒会", target="student council", priority=60,
        ),
        TranslationMemoryEntry(
            source="文化祭", target="school festival", priority=60,
        ),
        TranslationMemoryEntry(
            source="告白", target="confession", priority=50,
            notes="Love confession",
        ),
    ]

    # -- Regex rules ----------------------------------------------------
    ctx.regex_rules = [
        RegexRule(
            pattern=r"「[^」]*」",
            action="preserve",
            stage="pre",
            description="Preserve dialogue in Japanese quotation brackets 「」",
        ),
        RegexRule(
            pattern=r"（[^）]*）",
            action="preserve",
            stage="pre",
            description="Preserve inner thoughts in parentheses （）",
        ),
    ]

    # -- Formatting rules -----------------------------------------------
    ctx.formatting_rules = FormattingRules(
        preserve_honorifics=True,
        attack_uppercase=False,
        translate_sound_effects=True,
        preserve_brackets=True,
        language_style="neutral",
    )

    return profile
