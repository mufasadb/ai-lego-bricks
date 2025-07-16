# Text-to-Speech Service - Detailed Technical Documentation

## 🎵 Architecture Overview

The Text-to-Speech (TTS) Service provides a comprehensive voice synthesis solution with support for multiple providers, streaming capabilities, and advanced voice control features. It offers both simple text-to-speech conversion and sophisticated streaming pipelines for real-time applications.

### Core Components

```
TTS Service Ecosystem
├── TTSService (Main Service)
│   ├── Provider Abstraction Layer
│   ├── Voice Management System
│   ├── Audio Format Control
│   └── Quality Optimization
├── Provider Clients
│   ├── OpenAITTSClient (OpenAI TTS API)
│   ├── GoogleTTSClient (Google Cloud TTS)
│   ├── CoquiXTTSClient (Local Coqui-XTTS)
│   └── AzureTTSClient (Future Implementation)
├── Streaming Services
│   ├── StreamingTTSService (Sentence-level Streaming)
│   ├── RealtimeTTSPipeline (Live Audio Generation)
│   ├── LLMTTSIntegration (LLM + TTS Streaming)
│   └── AudioBufferManager (Buffer Management)
├── Voice Management
│   ├── VoiceSelector (Voice Discovery)
│   ├── VoiceCloning (Custom Voice Support)
│   ├── VoiceOptimization (Quality Enhancement)
│   └── EmotionControl (Emotional Expression)
└── Audio Processing
    ├── AudioPostProcessor (Enhancement)
    ├── FormatConverter (Multi-format Support)
    ├── QualityAnalyzer (Audio Quality Assessment)
    └── PerformanceOptimizer (Speed/Quality Balance)
```

## 🏗️ Core Service Implementation

### TTSService Architecture

```python
class TTSService:
    """
    Main TTS service providing unified text-to-speech interface.
    
    Design Principles:
    - Provider agnostic interface
    - Voice flexibility and control
    - Audio quality optimization
    - Streaming support
    - Performance monitoring
    """
    
    def __init__(self, client: TTSClient):
        """
        Initialize TTS service with enhanced capabilities.
        
        Initialization Features:
        - Client validation and setup
        - Voice discovery and caching
        - Audio format optimization
        - Performance tracking
        - Quality assessment
        """
        
        self.client = client
        self.config = client.config
        
        # Enhanced service capabilities
        self.voice_manager = VoiceManager(client)
        self.audio_processor = AudioPostProcessor()
        self.performance_tracker = TTSPerformanceTracker()
        self.quality_analyzer = TTSQualityAnalyzer()
        
        # Streaming capabilities
        self.streaming_service = StreamingTTSService(self)
        
        # Voice and audio caching
        self.voice_cache = {}
        self.audio_cache = {}
        self.cache_enabled = getattr(client.config, 'enable_cache', True)
        
        # Initialize voice discovery
        self._initialize_voice_discovery()
    
    def text_to_speech(
        self, 
        text: str, 
        voice: Optional[str] = None,
        output_path: Optional[str] = None, 
        **kwargs
    ) -> TTSResponse:
        """
        Convert text to speech with comprehensive processing pipeline.
        
        Processing Pipeline:
        1. Text validation and preprocessing
        2. Voice selection and optimization
        3. Audio generation with quality control
        4. Post-processing and enhancement
        5. Format conversion and output
        """
        
        import time
        start_time = time.time()
        
        try:
            # Stage 1: Text Validation and Preprocessing
            processed_text = self._preprocess_text(text)
            if not processed_text.valid:
                return TTSResponse(
                    success=False,
                    error_message=processed_text.error_message,
                    provider=self.config.provider.value
                )
            
            # Stage 2: Voice Selection and Optimization
            selected_voice = self._select_optimal_voice(voice, processed_text.content)
            
            # Stage 3: Cache Check
            if self.cache_enabled:
                cache_key = self._generate_cache_key(processed_text.content, selected_voice, kwargs)
                cached_response = self.audio_cache.get(cache_key)
                if cached_response:
                    cached_response.cached = True
                    return cached_response
            
            # Stage 4: Audio Generation
            generation_config = self._merge_generation_config(selected_voice, output_path, kwargs)
            response = self._execute_generation(processed_text.content, generation_config)
            
            # Stage 5: Post-processing and Enhancement
            enhanced_response = self._enhance_audio_response(
                response,
                processed_text,
                generation_config,
                time.time() - start_time
            )
            
            # Stage 6: Cache Storage
            if self.cache_enabled and enhanced_response.success:
                self.audio_cache[cache_key] = enhanced_response
            
            # Stage 7: Performance Tracking
            self.performance_tracker.record_generation(
                provider=self.config.provider.value,
                text_length=len(processed_text.content),
                duration=time.time() - start_time,
                success=enhanced_response.success,
                voice_used=selected_voice
            )
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"TTS processing failed: {e}")
            return TTSResponse(
                success=False,
                error_message=f"TTS processing failed: {str(e)}",
                provider=self.config.provider.value,
                processing_time=time.time() - start_time
            )
    
    def _preprocess_text(self, text: str) -> TextProcessingResult:
        """
        Comprehensive text preprocessing for optimal TTS generation.
        
        Text Processing:
        - Input validation and sanitization
        - Length and complexity checks
        - SSML tag validation
        - Pronunciation optimization
        - Emotional markup handling
        """
        
        # Basic validation
        if not text or not text.strip():
            return TextProcessingResult(
                valid=False,
                error_message="Text cannot be empty"
            )
        
        cleaned_text = text.strip()
        
        # Length validation
        max_length = getattr(self.config, 'max_text_length', 5000)
        if len(cleaned_text) > max_length:
            return TextProcessingResult(
                valid=False,
                error_message=f"Text length {len(cleaned_text)} exceeds maximum {max_length}"
            )
        
        # SSML validation and processing
        if self._contains_ssml(cleaned_text):
            ssml_result = self._validate_and_process_ssml(cleaned_text)
            if not ssml_result.valid:
                return TextProcessingResult(
                    valid=False,
                    error_message=f"Invalid SSML: {ssml_result.error_message}"
                )
            cleaned_text = ssml_result.processed_text
        
        # Text optimization for TTS
        optimized_text = self._optimize_text_for_tts(cleaned_text)
        
        return TextProcessingResult(
            valid=True,
            content=optimized_text,
            original_length=len(text),
            processed_length=len(optimized_text),
            contains_ssml=self._contains_ssml(optimized_text),
            estimated_duration=self._estimate_speech_duration(optimized_text)
        )
    
    def _optimize_text_for_tts(self, text: str) -> str:
        """
        Optimize text for better TTS pronunciation and naturalness.
        
        Optimization Techniques:
        - Number and abbreviation expansion
        - Punctuation normalization
        - Pronunciation hints
        - Pause insertion
        - Emotion markup
        """
        
        import re
        
        optimized = text
        
        # Expand common abbreviations
        abbreviation_map = {
            r'\bDr\.': 'Doctor',
            r'\bMr\.': 'Mister',
            r'\bMrs\.': 'Missus',
            r'\bMs\.': 'Miss',
            r'\betc\.': 'etcetera',
            r'\bi\.e\.': 'that is',
            r'\be\.g\.': 'for example',
            r'\bvs\.': 'versus',
            r'\bUSA\b': 'United States of America',
            r'\bUK\b': 'United Kingdom'
        }
        
        for pattern, replacement in abbreviation_map.items():
            optimized = re.sub(pattern, replacement, optimized, flags=re.IGNORECASE)
        
        # Expand numbers (basic implementation)
        def expand_numbers(match):
            number = match.group()
            try:
                if '.' in number:
                    # Handle decimals
                    return self._number_to_words_decimal(float(number))
                else:
                    # Handle integers
                    return self._number_to_words(int(number))
            except ValueError:
                return number
        
        # Replace standalone numbers
        optimized = re.sub(r'\b\d+\.?\d*\b', expand_numbers, optimized)
        
        # Add natural pauses
        optimized = re.sub(r'([.!?])\s+', r'\1 <break time="0.5s"/> ', optimized)
        optimized = re.sub(r'([,;])\s+', r'\1 <break time="0.3s"/> ', optimized)
        
        # Normalize whitespace
        optimized = re.sub(r'\s+', ' ', optimized).strip()
        
        return optimized
    
    def _select_optimal_voice(self, requested_voice: Optional[str], text: str) -> str:
        """
        Select optimal voice based on request, text content, and availability.
        
        Voice Selection Algorithm:
        1. Use explicitly requested voice if available
        2. Analyze text for optimal voice characteristics
        3. Check voice availability and quality
        4. Apply fallback strategy
        5. Optimize for provider capabilities
        """
        
        # Use requested voice if specified and available
        if requested_voice:
            available_voices = self.voice_manager.get_available_voices()
            if requested_voice in available_voices:
                return requested_voice
            else:
                logger.warning(f"Requested voice '{requested_voice}' not available, selecting alternative")
        
        # Analyze text for voice selection
        text_analysis = self._analyze_text_for_voice_selection(text)
        
        # Get optimal voice based on analysis
        optimal_voice = self.voice_manager.select_optimal_voice(
            text_characteristics=text_analysis,
            preference=requested_voice,
            provider_capabilities=self.client.get_capabilities()
        )
        
        return optimal_voice or self.config.voice or self.voice_manager.get_default_voice()
    
    def _analyze_text_for_voice_selection(self, text: str) -> TextCharacteristics:
        """
        Analyze text to determine optimal voice characteristics.
        
        Analysis Factors:
        - Text language and locale
        - Formality level
        - Emotional tone
        - Gender preferences (if applicable)
        - Age appropriateness
        """
        
        # Basic language detection
        detected_language = self._detect_language(text)
        
        # Formality analysis
        formality_score = self._analyze_formality(text)
        
        # Emotional tone analysis
        emotional_tone = self._analyze_emotional_tone(text)
        
        # Content type detection
        content_type = self._detect_content_type(text)
        
        return TextCharacteristics(
            language=detected_language,
            formality_level=formality_score,
            emotional_tone=emotional_tone,
            content_type=content_type,
            length=len(text),
            complexity=self._calculate_text_complexity(text)
        )
```

### Provider-Specific Implementations

#### OpenAI TTS Client

```python
class OpenAITTSClient(TTSClient):
    """
    OpenAI TTS client with advanced voice control and streaming support.
    
    Features:
    - High-quality neural voices
    - Multiple voice personalities
    - Streaming audio generation
    - Multiple output formats
    - Speed and pitch control
    """
    
    def __init__(self, config: TTSConfig, credential_manager: Optional['CredentialManager'] = None):
        """
        Initialize OpenAI TTS client.
        
        Initialization:
        - API key validation
        - Model and voice configuration
        - Format optimization
        - Streaming setup
        """
        
        super().__init__(config)
        self.credential_manager = credential_manager or default_credential_manager
        
        # API configuration
        self.api_key = self._get_api_key()
        self.client = self._initialize_openai_client()
        
        # Voice and model configuration
        self.model = config.model or "tts-1-hd"
        self.available_models = ["tts-1", "tts-1-hd"]
        
        # Voice configuration
        self.available_voices = {
            "alloy": {"gender": "neutral", "style": "balanced", "language": "en"},
            "echo": {"gender": "male", "style": "calm", "language": "en"},
            "fable": {"gender": "neutral", "style": "expressive", "language": "en"},
            "onyx": {"gender": "male", "style": "deep", "language": "en"},
            "nova": {"gender": "female", "style": "warm", "language": "en"},
            "shimmer": {"gender": "female", "style": "bright", "language": "en"}
        }
        
        # Format and quality settings
        self.optimal_formats = ["mp3", "opus", "aac", "flac"]
        self.streaming_supported = True
        
        # Validate configuration
        self._validate_configuration()
    
    def _initialize_openai_client(self):
        """Initialize OpenAI client with proper configuration."""
        
        try:
            import openai
            
            client = openai.OpenAI(
                api_key=self.api_key,
                timeout=60.0  # Generous timeout for audio generation
            )
            
            # Test client with a simple request
            self._test_client_connectivity(client)
            
            return client
            
        except Exception as e:
            logger.error(f"OpenAI TTS client initialization failed: {e}")
            raise RuntimeError(f"OpenAI TTS initialization failed: {e}")
    
    def text_to_speech(self, text: str, **kwargs) -> TTSResponse:
        """
        Generate speech using OpenAI TTS API.
        
        Generation Process:
        1. Build request parameters
        2. Make API call with retry logic
        3. Process audio response
        4. Save to file or return data
        5. Extract metadata
        """
        
        import time
        start_time = time.time()
        
        try:
            # Build request parameters
            request_params = self._build_openai_request(text, kwargs)
            
            # Make API call with retry logic
            response = self._call_with_retry(request_params)
            
            # Process response
            return self._process_openai_response(response, text, request_params, start_time)
            
        except Exception as e:
            logger.error(f"OpenAI TTS generation failed: {e}")
            return TTSResponse(
                success=False,
                error_message=f"OpenAI TTS generation failed: {str(e)}",
                provider="openai",
                processing_time=time.time() - start_time
            )
    
    def _build_openai_request(self, text: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build OpenAI TTS request parameters.
        
        Parameter Configuration:
        - Model selection and optimization
        - Voice selection and fallback
        - Speed control
        - Output format optimization
        """
        
        # Voice selection with validation
        voice = kwargs.get('voice', self.config.voice)
        if voice not in self.available_voices:
            voice = "alloy"  # Default fallback
            logger.warning(f"Invalid voice specified, using default: {voice}")
        
        # Model selection
        model = kwargs.get('model', self.model)
        if model not in self.available_models:
            model = "tts-1-hd"  # High quality default
        
        # Speed control
        speed = kwargs.get('speed', self.config.speed)
        speed = max(0.25, min(4.0, speed))  # OpenAI limits
        
        # Output format
        output_format = kwargs.get('output_format', self.config.output_format.value)
        if output_format not in self.optimal_formats:
            output_format = "mp3"  # Default format
        
        return {
            "model": model,
            "input": text,
            "voice": voice,
            "speed": speed,
            "response_format": output_format
        }
    
    def _call_with_retry(self, request_params: Dict[str, Any], max_retries: int = 3) -> bytes:
        """
        Make API call with exponential backoff retry logic.
        
        Retry Strategy:
        - Exponential backoff
        - Rate limit handling
        - Error differentiation
        - Timeout management
        """
        
        import time
        import openai
        
        for attempt in range(max_retries + 1):
            try:
                response = self.client.audio.speech.create(**request_params)
                return response.content
                
            except openai.RateLimitError as e:
                if attempt < max_retries:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"OpenAI rate limit hit, waiting {wait_time:.2f}s before retry {attempt + 1}")
                    time.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError(f"OpenAI rate limit exceeded after {max_retries} retries")
            
            except openai.APITimeoutError as e:
                if attempt < max_retries:
                    wait_time = (2 ** attempt) * 2  # Longer wait for timeouts
                    logger.warning(f"OpenAI API timeout, waiting {wait_time:.2f}s before retry {attempt + 1}")
                    time.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError(f"OpenAI API timeout after {max_retries} retries")
            
            except Exception as e:
                if attempt < max_retries and "server_error" in str(e).lower():
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"OpenAI server error, waiting {wait_time:.2f}s before retry {attempt + 1}")
                    time.sleep(wait_time)
                    continue
                else:
                    raise e
    
    def _process_openai_response(
        self, 
        audio_content: bytes, 
        text: str, 
        request_params: Dict[str, Any],
        start_time: float
    ) -> TTSResponse:
        """
        Process OpenAI API response into TTSResponse format.
        
        Response Processing:
        - Save audio to file
        - Extract metadata
        - Calculate quality metrics
        - Estimate duration
        """
        
        import tempfile
        import os
        
        try:
            # Generate output filename
            output_path = self._generate_output_path(text, request_params)
            
            # Save audio content
            with open(output_path, 'wb') as audio_file:
                audio_file.write(audio_content)
            
            # Calculate audio metadata
            audio_info = self._analyze_generated_audio(output_path)
            
            # Calculate processing metrics
            processing_time = time.time() - start_time
            estimated_duration = len(text.split()) * 0.6  # Rough estimate: ~100 WPM
            
            return TTSResponse(
                success=True,
                audio_file_path=output_path,
                duration_ms=int(audio_info.duration * 1000),
                voice_used=request_params['voice'],
                model_used=request_params['model'],
                provider="openai",
                format_used=request_params['response_format'],
                processing_time=processing_time,
                text_length=len(text),
                estimated_speaking_rate=len(text.split()) / max(audio_info.duration / 60, 0.1),  # WPM
                file_size_bytes=len(audio_content),
                quality_metrics={
                    'sample_rate': audio_info.sample_rate,
                    'bit_depth': audio_info.bit_depth,
                    'channels': audio_info.channels,
                    'bitrate': audio_info.bitrate
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI response processing failed: {e}")
            return TTSResponse(
                success=False,
                error_message=f"Response processing failed: {str(e)}",
                provider="openai",
                processing_time=time.time() - start_time
            )
    
    def stream_text_to_speech(self, text: str, **kwargs) -> Iterator[bytes]:
        """
        Stream text-to-speech generation for real-time applications.
        
        Streaming Process:
        - Chunk text into sentences
        - Generate audio for each chunk
        - Yield audio data as available
        - Handle buffering and synchronization
        """
        
        try:
            # Split text into streaming chunks
            text_chunks = self._split_text_for_streaming(text)
            
            for chunk in text_chunks:
                if not chunk.strip():
                    continue
                
                # Build request for chunk
                request_params = self._build_openai_request(chunk, kwargs)
                
                # Generate audio for chunk
                try:
                    audio_content = self._call_with_retry(request_params)
                    yield audio_content
                except Exception as e:
                    logger.error(f"Streaming chunk failed: {e}")
                    # Continue with next chunk
                    continue
                    
        except Exception as e:
            logger.error(f"OpenAI streaming failed: {e}")
            raise RuntimeError(f"Streaming generation failed: {str(e)}")
```

#### Google Cloud TTS Client

```python
class GoogleTTSClient(TTSClient):
    """
    Google Cloud TTS client with advanced SSML support and voice selection.
    
    Features:
    - 200+ voices in 40+ languages
    - WaveNet and Neural2 voices
    - SSML markup support
    - Custom voice training
    - Audio effects and processing
    """
    
    def __init__(self, config: TTSConfig, credential_manager: Optional['CredentialManager'] = None):
        """
        Initialize Google TTS client.
        
        Initialization:
        - Credentials validation
        - Voice discovery
        - Language configuration
        - SSML support setup
        """
        
        super().__init__(config)
        self.credential_manager = credential_manager or default_credential_manager
        
        # Initialize Google TTS client
        self.client = self._initialize_google_client()
        
        # Voice and language configuration
        self.voice_discovery = GoogleVoiceDiscovery(self.client)
        self.available_voices = self.voice_discovery.discover_voices()
        
        # SSML and audio effects
        self.ssml_processor = SSMLProcessor()
        self.audio_effects = GoogleAudioEffects()
        
        # Format and quality settings
        self.optimal_formats = ["mp3", "wav", "ogg"]
        self.supported_encodings = [
            "LINEAR16", "MP3", "OGG_OPUS", "MULAW", "ALAW"
        ]
        
        # Voice quality tiers
        self.voice_quality_tiers = {
            "standard": ["standard"],
            "wavenet": ["wavenet"],
            "neural2": ["neural2"],
            "studio": ["studio"]
        }
    
    def _initialize_google_client(self):
        """Initialize Google Cloud TTS client with authentication."""
        
        try:
            from google.cloud import texttospeech
            
            # Check for explicit credentials
            credentials_path = self.credential_manager.get_credential("GOOGLE_APPLICATION_CREDENTIALS")
            
            if credentials_path and os.path.exists(credentials_path):
                client = texttospeech.TextToSpeechClient.from_service_account_json(credentials_path)
                logger.info("✓ Google TTS initialized with service account")
            else:
                client = texttospeech.TextToSpeechClient()
                logger.info("✓ Google TTS initialized with default credentials")
            
            return client
            
        except Exception as e:
            logger.error(f"Google TTS client initialization failed: {e}")
            raise RuntimeError(f"Google TTS initialization failed: {e}")
    
    def text_to_speech(self, text: str, **kwargs) -> TTSResponse:
        """
        Generate speech using Google Cloud TTS.
        
        Advanced Features:
        - SSML processing and validation
        - Voice selection optimization
        - Audio effects application
        - Quality tier selection
        """
        
        import time
        start_time = time.time()
        
        try:
            # Process and validate SSML
            processed_input = self._process_text_input(text, kwargs)
            
            # Select optimal voice
            voice_selection = self._select_google_voice(kwargs, processed_input)
            
            # Build synthesis request
            synthesis_request = self._build_google_synthesis_request(
                processed_input, 
                voice_selection, 
                kwargs
            )
            
            # Execute synthesis
            response = self.client.synthesize_speech(request=synthesis_request)
            
            # Process and enhance response
            return self._process_google_response(
                response, 
                text, 
                voice_selection, 
                synthesis_request,
                start_time
            )
            
        except Exception as e:
            logger.error(f"Google TTS generation failed: {e}")
            return TTSResponse(
                success=False,
                error_message=f"Google TTS generation failed: {str(e)}",
                provider="google",
                processing_time=time.time() - start_time
            )
    
    def _process_text_input(self, text: str, kwargs: Dict[str, Any]) -> ProcessedTextInput:
        """
        Process and validate text input for Google TTS.
        
        Processing Features:
        - SSML validation and enhancement
        - Pronunciation optimization
        - Language detection
        - Content analysis
        """
        
        # Check if input is SSML
        is_ssml = self._is_ssml_input(text)
        
        if is_ssml:
            # Validate and process SSML
            ssml_result = self.ssml_processor.validate_and_enhance(text)
            if not ssml_result.valid:
                raise ValueError(f"Invalid SSML: {ssml_result.error_message}")
            
            processed_text = ssml_result.enhanced_ssml
            input_type = "ssml"
        else:
            # Process plain text and optionally convert to SSML
            if kwargs.get('enhance_with_ssml', False):
                processed_text = self.ssml_processor.text_to_ssml(text, kwargs)
                input_type = "ssml"
            else:
                processed_text = text
                input_type = "text"
        
        # Language detection
        detected_language = self._detect_language(processed_text)
        
        return ProcessedTextInput(
            content=processed_text,
            input_type=input_type,
            language=detected_language,
            estimated_duration=self._estimate_duration(processed_text),
            complexity_score=self._calculate_complexity(processed_text)
        )
    
    def _select_google_voice(self, kwargs: Dict[str, Any], processed_input: ProcessedTextInput) -> GoogleVoiceSelection:
        """
        Select optimal Google voice based on requirements and content.
        
        Voice Selection Algorithm:
        - Language compatibility check
        - Quality tier preference
        - Gender and age preferences
        - Voice characteristic matching
        - Availability validation
        """
        
        # Extract voice preferences
        requested_voice = kwargs.get('voice')
        language_code = kwargs.get('language_code', processed_input.language)
        gender = kwargs.get('gender')
        quality_tier = kwargs.get('quality_tier', 'wavenet')
        
        # Filter available voices
        compatible_voices = self.voice_discovery.find_compatible_voices(
            language_code=language_code,
            gender=gender,
            quality_tier=quality_tier,
            content_characteristics=processed_input
        )
        
        if not compatible_voices:
            # Fallback to any available voice for the language
            compatible_voices = self.voice_discovery.find_compatible_voices(
                language_code=language_code
            )
        
        if not compatible_voices:
            raise ValueError(f"No compatible voices found for language: {language_code}")
        
        # Select best voice
        if requested_voice and requested_voice in [v.name for v in compatible_voices]:
            selected_voice = next(v for v in compatible_voices if v.name == requested_voice)
        else:
            # Use ranking algorithm
            selected_voice = self._rank_and_select_voice(compatible_voices, processed_input, kwargs)
        
        return GoogleVoiceSelection(
            voice=selected_voice,
            language_code=language_code,
            selection_reason=f"Optimized for {quality_tier} quality",
            alternatives=[v.name for v in compatible_voices[:3]]  # Top 3 alternatives
        )
    
    def _build_google_synthesis_request(
        self,
        processed_input: ProcessedTextInput,
        voice_selection: GoogleVoiceSelection,
        kwargs: Dict[str, Any]
    ) -> 'texttospeech.SynthesizeSpeechRequest':
        """
        Build Google Cloud TTS synthesis request.
        
        Request Configuration:
        - Input text/SSML formatting
        - Voice and language configuration
        - Audio encoding and quality
        - Effects and processing options
        """
        
        from google.cloud import texttospeech
        
        # Configure synthesis input
        if processed_input.input_type == "ssml":
            synthesis_input = texttospeech.SynthesisInput(ssml=processed_input.content)
        else:
            synthesis_input = texttospeech.SynthesisInput(text=processed_input.content)
        
        # Configure voice
        voice = texttospeech.VoiceSelectionParams(
            language_code=voice_selection.language_code,
            name=voice_selection.voice.name,
            ssml_gender=self._map_gender_to_ssml(voice_selection.voice.gender)
        )
        
        # Configure audio
        audio_encoding = self._select_audio_encoding(kwargs)
        audio_config = texttospeech.AudioConfig(
            audio_encoding=audio_encoding,
            sample_rate_hertz=kwargs.get('sample_rate', 24000),
            speaking_rate=kwargs.get('speaking_rate', 1.0),
            pitch=kwargs.get('pitch', 0.0),
            volume_gain_db=kwargs.get('volume_gain_db', 0.0)
        )
        
        # Add audio effects if specified
        effects = kwargs.get('audio_effects', [])
        if effects:
            audio_config.effects_profile_id = effects
        
        return texttospeech.SynthesizeSpeechRequest(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
```

## 🎵 Streaming TTS Implementation

### Real-time Streaming Service

```python
class StreamingTTSService:
    """
    Advanced streaming TTS service for real-time applications.
    
    Features:
    - Sentence-level streaming
    - Buffer management
    - Quality optimization
    - Latency minimization
    - Error recovery
    """
    
    def __init__(
        self, 
        tts_service: TTSService,
        sentence_buffer_size: int = 3,
        max_buffer_time: float = 2.0,
        quality_mode: str = "balanced"
    ):
        """
        Initialize streaming TTS service.
        
        Configuration:
        - Base TTS service
        - Buffer management settings
        - Quality vs latency balance
        - Error handling preferences
        """
        
        self.tts_service = tts_service
        self.sentence_buffer_size = sentence_buffer_size
        self.max_buffer_time = max_buffer_time
        self.quality_mode = quality_mode
        
        # Streaming state management
        self.sentence_buffer = []
        self.buffer_lock = threading.Lock()
        self.generation_queue = queue.Queue()
        self.output_queue = queue.Queue()
        
        # Performance tracking
        self.streaming_metrics = {
            'sentences_processed': 0,
            'total_latency': 0.0,
            'buffer_overflows': 0,
            'generation_errors': 0
        }
        
        # Worker threads
        self.generation_thread = None
        self.is_streaming = False
    
    def stream_text_to_audio(
        self, 
        text_generator: Iterator[str],
        **kwargs
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream text chunks to audio with real-time processing.
        
        Streaming Pipeline:
        1. Receive text chunks from generator
        2. Buffer sentences for optimal processing
        3. Generate audio asynchronously
        4. Yield audio data as available
        5. Handle errors and recovery
        """
        
        try:
            # Start streaming threads
            self._start_streaming_threads(**kwargs)
            
            # Process text chunks
            for text_chunk in text_generator:
                self._process_text_chunk(text_chunk)
                
                # Yield available audio
                while not self.output_queue.empty():
                    try:
                        audio_data = self.output_queue.get_nowait()
                        yield audio_data
                    except queue.Empty:
                        break
            
            # Flush remaining buffers
            self._flush_buffers()
            
            # Yield final audio
            while not self.output_queue.empty():
                try:
                    audio_data = self.output_queue.get_nowait()
                    yield audio_data
                except queue.Empty:
                    break
                    
        finally:
            # Clean up streaming threads
            self._stop_streaming_threads()
    
    def _start_streaming_threads(self, **kwargs):
        """Start background threads for audio generation."""
        
        self.is_streaming = True
        
        # Start audio generation thread
        self.generation_thread = threading.Thread(
            target=self._audio_generation_worker,
            args=(kwargs,)
        )
        self.generation_thread.start()
    
    def _audio_generation_worker(self, generation_kwargs: Dict[str, Any]):
        """
        Background worker for audio generation.
        
        Worker Process:
        - Monitor sentence buffer
        - Trigger audio generation
        - Handle generation errors
        - Queue audio output
        """
        
        while self.is_streaming:
            try:
                # Wait for sentences to generate
                sentence_batch = self.generation_queue.get(timeout=0.1)
                
                if sentence_batch is None:  # Shutdown signal
                    break
                
                # Generate audio for sentence batch
                start_time = time.time()
                
                try:
                    audio_response = self.tts_service.text_to_speech(
                        text=sentence_batch['text'],
                        **{**generation_kwargs, **sentence_batch.get('overrides', {})}
                    )
                    
                    if audio_response.success:
                        # Queue successful audio
                        self.output_queue.put({
                            'status': 'audio_generated',
                            'audio_file': audio_response.audio_file_path,
                            'duration_ms': audio_response.duration_ms,
                            'sentence_count': sentence_batch['sentence_count'],
                            'generation_time': time.time() - start_time,
                            'sequence_number': sentence_batch['sequence_number']
                        })
                        
                        # Update metrics
                        self.streaming_metrics['sentences_processed'] += sentence_batch['sentence_count']
                        self.streaming_metrics['total_latency'] += time.time() - start_time
                    else:
                        # Handle generation error
                        self.output_queue.put({
                            'status': 'generation_error',
                            'error_message': audio_response.error_message,
                            'sentence_count': sentence_batch['sentence_count'],
                            'sequence_number': sentence_batch['sequence_number']
                        })
                        
                        self.streaming_metrics['generation_errors'] += 1
                        
                except Exception as e:
                    logger.error(f"Audio generation failed: {e}")
                    self.streaming_metrics['generation_errors'] += 1
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Audio generation worker error: {e}")
                continue
    
    def _process_text_chunk(self, text_chunk: str):
        """
        Process incoming text chunk for streaming.
        
        Processing Steps:
        - Split into sentences
        - Add to sentence buffer
        - Trigger generation when buffer ready
        - Handle buffer overflow
        """
        
        # Split chunk into sentences
        sentences = self._split_into_sentences(text_chunk)
        
        with self.buffer_lock:
            # Add sentences to buffer
            self.sentence_buffer.extend(sentences)
            
            # Check if buffer is ready for processing
            if (len(self.sentence_buffer) >= self.sentence_buffer_size or
                self._buffer_time_exceeded()):
                
                # Create sentence batch for generation
                batch_text = ' '.join(self.sentence_buffer)
                sentence_batch = {
                    'text': batch_text,
                    'sentence_count': len(self.sentence_buffer),
                    'sequence_number': self.streaming_metrics['sentences_processed'],
                    'timestamp': time.time()
                }
                
                # Queue for generation
                try:
                    self.generation_queue.put_nowait(sentence_batch)
                    
                    # Clear buffer
                    self.sentence_buffer = []
                    
                except queue.Full:
                    # Handle buffer overflow
                    self.streaming_metrics['buffer_overflows'] += 1
                    logger.warning("Generation queue full, dropping sentence batch")
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences for streaming.
        
        Sentence Splitting:
        - Handle abbreviations
        - Preserve sentence boundaries
        - Optimize for TTS processing
        """
        
        import re
        
        # Simple sentence splitting (can be enhanced)
        sentence_endings = r'[.!?]+\s+'
        sentences = re.split(sentence_endings, text.strip())
        
        # Filter empty sentences and clean up
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 3:  # Minimum meaningful sentence
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
```

### LLM Integration Pipeline

```python
class LLMTTSIntegrationPipeline:
    """
    Integrated LLM + TTS pipeline for conversational AI.
    
    Features:
    - Real-time LLM streaming
    - Concurrent TTS generation
    - Voice response streaming
    - Context preservation
    - Error recovery
    """
    
    def __init__(
        self,
        llm_service,
        tts_service: TTSService,
        response_buffer_size: int = 2,
        concurrent_generations: int = 3
    ):
        """
        Initialize LLM-TTS integration pipeline.
        
        Pipeline Configuration:
        - LLM service for text generation
        - TTS service for voice synthesis
        - Buffer management settings
        - Concurrency control
        """
        
        self.llm_service = llm_service
        self.tts_service = tts_service
        self.response_buffer_size = response_buffer_size
        self.concurrent_generations = concurrent_generations
        
        # Pipeline state
        self.is_active = False
        self.conversation_context = []
        
        # Processing queues
        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        
        # Worker pools
        self.tts_workers = []
        self.generation_semaphore = threading.Semaphore(concurrent_generations)
    
    def stream_conversation_response(
        self,
        user_input: str,
        conversation_context: Optional[List] = None,
        voice_config: Optional[Dict] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream conversational response with voice synthesis.
        
        Conversation Pipeline:
        1. Process user input with LLM
        2. Stream LLM response tokens
        3. Buffer tokens into sentences
        4. Generate audio concurrently
        5. Stream audio as available
        """
        
        try:
            # Start pipeline
            self._start_pipeline()
            
            # Update conversation context
            if conversation_context:
                self.conversation_context = conversation_context
            
            # Start LLM streaming
            llm_stream = self.llm_service.stream_chat(
                user_input, 
                self.conversation_context
            )
            
            # Process LLM stream
            sentence_buffer = ""
            
            for token in llm_stream:
                sentence_buffer += token
                
                # Check for sentence completion
                if self._is_sentence_complete(sentence_buffer):
                    # Queue sentence for TTS
                    self._queue_sentence_for_tts(sentence_buffer.strip(), voice_config)
                    sentence_buffer = ""
                
                # Yield available audio
                while not self.audio_queue.empty():
                    try:
                        audio_data = self.audio_queue.get_nowait()
                        yield audio_data
                    except queue.Empty:
                        break
            
            # Process remaining text
            if sentence_buffer.strip():
                self._queue_sentence_for_tts(sentence_buffer.strip(), voice_config)
            
            # Wait for remaining audio
            while self._has_pending_generations():
                try:
                    audio_data = self.audio_queue.get(timeout=1.0)
                    yield audio_data
                except queue.Empty:
                    break
                    
        finally:
            # Clean up pipeline
            self._stop_pipeline()
    
    def _queue_sentence_for_tts(self, sentence: str, voice_config: Optional[Dict]):
        """
        Queue sentence for TTS generation.
        
        Queueing Process:
        - Validate sentence content
        - Apply voice configuration
        - Submit to worker pool
        - Handle queue management
        """
        
        if not sentence.strip():
            return
        
        # Create TTS task
        tts_task = {
            'text': sentence,
            'voice_config': voice_config or {},
            'timestamp': time.time(),
            'sequence_number': len(self.conversation_context)
        }
        
        # Submit to worker pool
        worker_thread = threading.Thread(
            target=self._process_tts_task,
            args=(tts_task,)
        )
        worker_thread.start()
        self.tts_workers.append(worker_thread)
    
    def _process_tts_task(self, tts_task: Dict[str, Any]):
        """
        Process individual TTS task.
        
        Task Processing:
        - Acquire generation semaphore
        - Generate audio
        - Queue audio result
        - Handle errors
        """
        
        with self.generation_semaphore:
            try:
                # Generate audio
                response = self.tts_service.text_to_speech(
                    text=tts_task['text'],
                    **tts_task['voice_config']
                )
                
                if response.success:
                    # Queue successful audio
                    self.audio_queue.put({
                        'status': 'audio_ready',
                        'audio_file': response.audio_file_path,
                        'text': tts_task['text'],
                        'duration_ms': response.duration_ms,
                        'sequence_number': tts_task['sequence_number'],
                        'generation_time': response.processing_time
                    })
                else:
                    # Queue error
                    self.audio_queue.put({
                        'status': 'audio_error',
                        'error_message': response.error_message,
                        'text': tts_task['text'],
                        'sequence_number': tts_task['sequence_number']
                    })
                    
            except Exception as e:
                logger.error(f"TTS task processing failed: {e}")
                self.audio_queue.put({
                    'status': 'audio_error',
                    'error_message': str(e),
                    'text': tts_task['text'],
                    'sequence_number': tts_task['sequence_number']
                })
```

This comprehensive documentation provides deep technical insight into the TTS service architecture, provider implementations, streaming capabilities, and advanced integration patterns for real-time voice applications.