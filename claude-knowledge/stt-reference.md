# Speech-to-Text Service - Detailed Technical Documentation

## 🎤 Architecture Overview

The Speech-to-Text (STT) Service provides a unified interface for converting speech to text across multiple providers. It supports advanced features like word-level timestamps, speaker diarization, language detection, and multi-format audio processing with automatic provider selection and fallback mechanisms.

### Core Components

```
STT Service Ecosystem
├── STTService (Main Service)
│   ├── Provider Abstraction Layer
│   ├── Audio Format Validation
│   ├── Language Detection
│   └── Response Processing
├── Provider Clients
│   ├── FasterWhisperClient (Local Processing)
│   ├── GoogleSpeechClient (Cloud Processing)
│   ├── OpenAIWhisperClient (API Processing)
│   └── Azure CognitiveClient (Future)
├── Configuration System
│   ├── STTConfig (Provider Settings)
│   ├── Audio Format Management
│   ├── Language Configuration
│   └── Quality Settings
├── Response Processing
│   ├── STTResponse (Structured Results)
│   ├── WordTimestamp (Word-level Timing)
│   ├── SpeakerSegment (Speaker Diarization)
│   └── Confidence Scoring
└── Factory Pattern
    ├── Auto-provider Detection
    ├── Service Creation
    └── Configuration Management
```

## 🏗️ Core Service Implementation

### STTService Architecture

```python
class STTService:
    """
    Main STT service providing unified speech-to-text interface.
    
    Design Principles:
    - Provider agnostic interface
    - Comprehensive error handling
    - Audio format validation
    - Quality assessment
    - Performance optimization
    """
    
    def __init__(self, client: STTClient):
        """
        Initialize STT service with client and enhanced capabilities.
        
        Initialization Features:
        - Client validation and setup
        - Audio format detection
        - Performance metrics tracking
        - Error handling configuration
        """
        
        self.client = client
        self.config = client.config
        
        # Enhanced service capabilities
        self.audio_validator = AudioValidator()
        self.performance_tracker = PerformanceTracker()
        self.quality_assessor = TranscriptionQualityAssessor()
        
        # Cache for repeated requests
        self.transcription_cache = {}
        self.cache_enabled = getattr(client.config, 'enable_cache', False)
        
        # Supported audio formats by provider
        self.format_support_matrix = self._build_format_support_matrix()
    
    def speech_to_text(
        self, 
        audio_file_path: str, 
        language: Optional[str] = None,
        **kwargs
    ) -> STTResponse:
        """
        Convert speech to text with comprehensive processing pipeline.
        
        Processing Pipeline:
        1. Input validation and format detection
        2. Audio preprocessing and optimization
        3. Provider-specific transcription
        4. Post-processing and quality assessment
        5. Response structuring and metadata addition
        """
        
        import time
        start_time = time.time()
        
        try:
            # Stage 1: Input Validation
            validation_result = self._validate_audio_input(audio_file_path)
            if not validation_result.valid:
                return STTResponse(
                    success=False,
                    error_message=validation_result.error_message,
                    provider=self.config.provider.value
                )
            
            # Stage 2: Cache Check
            if self.cache_enabled:
                cache_key = self._generate_cache_key(audio_file_path, language, kwargs)
                cached_response = self.transcription_cache.get(cache_key)
                if cached_response:
                    cached_response.cached = True
                    return cached_response
            
            # Stage 3: Audio Preprocessing
            preprocessed_path = self._preprocess_audio(audio_file_path, validation_result)
            
            # Stage 4: Provider Transcription
            transcription_config = self._merge_transcription_config(language, kwargs)
            response = self._execute_transcription(preprocessed_path, transcription_config)
            
            # Stage 5: Post-processing
            enhanced_response = self._enhance_transcription_response(
                response, 
                audio_file_path,
                time.time() - start_time
            )
            
            # Stage 6: Cache Storage
            if self.cache_enabled and enhanced_response.success:
                self.transcription_cache[cache_key] = enhanced_response
            
            # Stage 7: Performance Tracking
            self.performance_tracker.record_transcription(
                provider=self.config.provider.value,
                duration=time.time() - start_time,
                success=enhanced_response.success,
                audio_length=validation_result.duration
            )
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"STT processing failed: {e}")
            return STTResponse(
                success=False,
                error_message=f"STT processing failed: {str(e)}",
                provider=self.config.provider.value,
                processing_time=time.time() - start_time
            )
    
    def _validate_audio_input(self, audio_file_path: str) -> AudioValidationResult:
        """
        Comprehensive audio input validation.
        
        Validation Checks:
        - File existence and accessibility
        - Audio format compatibility
        - File size and duration limits
        - Audio quality assessment
        - Encoding validation
        """
        
        # Basic file checks
        if not audio_file_path or not os.path.isfile(audio_file_path):
            return AudioValidationResult(
                valid=False,
                error_message=f"Audio file not found: {audio_file_path}"
            )
        
        try:
            # Audio format and metadata extraction
            audio_info = self.audio_validator.analyze_audio_file(audio_file_path)
            
            # Format compatibility check
            if not self._is_format_supported(audio_info.format):
                return AudioValidationResult(
                    valid=False,
                    error_message=f"Unsupported audio format: {audio_info.format}"
                )
            
            # Duration limits check
            if audio_info.duration > self.config.max_audio_duration:
                return AudioValidationResult(
                    valid=False,
                    error_message=f"Audio duration {audio_info.duration}s exceeds limit {self.config.max_audio_duration}s"
                )
            
            # Quality assessment
            quality_check = self.audio_validator.assess_audio_quality(audio_file_path)
            if quality_check.quality_score < 0.3:
                logger.warning(f"Low audio quality detected: {quality_check.quality_score}")
            
            return AudioValidationResult(
                valid=True,
                audio_info=audio_info,
                quality_assessment=quality_check
            )
            
        except Exception as e:
            return AudioValidationResult(
                valid=False,
                error_message=f"Audio validation failed: {str(e)}"
            )
    
    def _preprocess_audio(self, audio_file_path: str, validation_result: AudioValidationResult) -> str:
        """
        Preprocess audio for optimal transcription quality.
        
        Preprocessing Steps:
        - Format conversion if needed
        - Audio normalization
        - Noise reduction (optional)
        - Sample rate optimization
        - Channel configuration
        """
        
        audio_info = validation_result.audio_info
        
        # Check if preprocessing is needed
        preprocessing_needed = (
            audio_info.format not in self.client.optimal_formats or
            audio_info.sample_rate != self.client.optimal_sample_rate or
            validation_result.quality_assessment.needs_enhancement
        )
        
        if not preprocessing_needed:
            return audio_file_path
        
        try:
            # Create preprocessed version
            preprocessed_path = self._create_preprocessed_audio(
                audio_file_path, 
                audio_info, 
                validation_result.quality_assessment
            )
            
            return preprocessed_path
            
        except Exception as e:
            logger.warning(f"Audio preprocessing failed: {e}, using original file")
            return audio_file_path
    
    def _create_preprocessed_audio(
        self, 
        original_path: str, 
        audio_info: AudioInfo, 
        quality_assessment: AudioQualityAssessment
    ) -> str:
        """
        Create optimized version of audio file.
        
        Optimization Techniques:
        - Format conversion to optimal format
        - Sample rate conversion
        - Audio normalization
        - Noise reduction
        - Channel mixing/separation
        """
        
        import tempfile
        import subprocess
        
        # Create temporary file for preprocessed audio
        temp_dir = tempfile.gettempdir()
        temp_filename = f"stt_preprocessed_{int(time.time())}.wav"
        preprocessed_path = os.path.join(temp_dir, temp_filename)
        
        # Build FFmpeg command for audio processing
        ffmpeg_cmd = self._build_ffmpeg_command(
            original_path,
            preprocessed_path,
            audio_info,
            quality_assessment
        )
        
        # Execute preprocessing
        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            return preprocessed_path
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg preprocessing failed: {e}")
            raise RuntimeError(f"Audio preprocessing failed: {e}")
    
    def _build_ffmpeg_command(
        self,
        input_path: str,
        output_path: str,
        audio_info: AudioInfo,
        quality_assessment: AudioQualityAssessment
    ) -> List[str]:
        """
        Build FFmpeg command for audio optimization.
        
        Command Building:
        - Input/output file specification
        - Format conversion parameters
        - Audio filters application
        - Quality enhancement options
        """
        
        cmd = ['ffmpeg', '-i', input_path]
        
        # Audio codec and format
        cmd.extend(['-acodec', 'pcm_s16le'])
        
        # Sample rate optimization
        optimal_rate = self.client.optimal_sample_rate
        if audio_info.sample_rate != optimal_rate:
            cmd.extend(['-ar', str(optimal_rate)])
        
        # Channel configuration
        if audio_info.channels > 1 and not self.config.enable_speaker_diarization:
            cmd.extend(['-ac', '1'])  # Convert to mono
        
        # Audio filters
        filters = []
        
        # Normalization
        if quality_assessment.needs_normalization:
            filters.append('loudnorm')
        
        # Noise reduction
        if quality_assessment.noise_level > 0.3:
            filters.append('afftdn=nf=-25')  # Noise reduction
        
        # High-pass filter for speech
        filters.append('highpass=f=80')
        
        # Low-pass filter to remove high-frequency noise
        filters.append('lowpass=f=8000')
        
        if filters:
            cmd.extend(['-af', ','.join(filters)])
        
        # Output file
        cmd.extend(['-y', output_path])  # -y to overwrite
        
        return cmd
```

### Provider-Specific Implementations

#### Faster Whisper Client (Local Processing)

```python
class FasterWhisperClient(STTClient):
    """
    Faster Whisper client for local speech-to-text processing.
    
    Features:
    - Local processing (no API calls)
    - Multiple model sizes
    - GPU acceleration support
    - Word-level timestamps
    - Language detection
    - Batch processing
    """
    
    def __init__(self, config: STTConfig, credential_manager: Optional['CredentialManager'] = None):
        """
        Initialize Faster Whisper client.
        
        Initialization:
        - Server connection validation
        - Model configuration
        - Performance optimization
        - GPU detection
        """
        
        super().__init__(config)
        self.credential_manager = credential_manager or default_credential_manager
        
        # Server configuration
        self.server_url = self._get_server_url()
        self.timeout = config.timeout or 120
        
        # Model configuration
        self.model = config.model or "base"
        self.available_models = ["tiny", "base", "small", "medium", "large"]
        
        # Performance settings
        self.optimal_formats = ["wav", "mp3", "flac"]
        self.optimal_sample_rate = 16000
        
        # Validate server availability
        self._validate_server_connection()
    
    def _get_server_url(self) -> str:
        """Get Faster Whisper server URL with fallback."""
        
        url = self.credential_manager.get_credential(
            "FASTER_WHISPER_URL", 
            "http://localhost:10300"
        )
        
        # Ensure URL format
        if not url.startswith(('http://', 'https://')):
            url = f"http://{url}"
        
        return url.rstrip('/')
    
    def _validate_server_connection(self):
        """Validate connection to Faster Whisper server."""
        
        try:
            import httpx
            
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.server_url}/health")
                
                if response.status_code == 200:
                    logger.info(f"✓ Faster Whisper server available at {self.server_url}")
                else:
                    raise RuntimeError(f"Server health check failed: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Faster Whisper server not available: {e}")
            raise RuntimeError(f"Faster Whisper server not available at {self.server_url}: {e}")
    
    def speech_to_text(self, audio_file_path: str, **kwargs) -> STTResponse:
        """
        Transcribe audio using Faster Whisper server.
        
        Transcription Process:
        1. Prepare multipart request
        2. Send to Faster Whisper server
        3. Parse detailed response
        4. Extract timestamps and segments
        5. Format as STTResponse
        """
        
        import httpx
        import json
        
        try:
            # Prepare request parameters
            transcription_params = self._build_transcription_params(kwargs)
            
            # Prepare file upload
            with open(audio_file_path, 'rb') as audio_file:
                files = {'audio': audio_file}
                data = transcription_params
                
                # Make request to Faster Whisper server
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(
                        f"{self.server_url}/transcribe",
                        files=files,
                        data=data
                    )
                    
                    response.raise_for_status()
                    result = response.json()
            
            # Parse response
            return self._parse_faster_whisper_response(result, audio_file_path)
            
        except httpx.TimeoutException:
            return STTResponse(
                success=False,
                error_message="Faster Whisper request timed out",
                provider="faster_whisper"
            )
        except httpx.HTTPStatusError as e:
            return STTResponse(
                success=False,
                error_message=f"Faster Whisper API error: {e.response.status_code}",
                provider="faster_whisper"
            )
        except Exception as e:
            return STTResponse(
                success=False,
                error_message=f"Faster Whisper transcription failed: {str(e)}",
                provider="faster_whisper"
            )
    
    def _build_transcription_params(self, kwargs: Dict[str, Any]) -> Dict[str, str]:
        """
        Build transcription parameters for Faster Whisper.
        
        Parameter Mapping:
        - Model selection
        - Language configuration
        - Output options
        - Quality settings
        """
        
        params = {
            'model': kwargs.get('model', self.model),
            'language': kwargs.get('language', self.config.language or 'auto'),
            'task': 'transcribe',  # vs translate
            'word_timestamps': str(kwargs.get('enable_word_timestamps', 
                                           self.config.enable_word_timestamps)).lower(),
            'output_format': 'json',
            'temperature': str(kwargs.get('temperature', 0.0)),
            'best_of': str(kwargs.get('best_of', 1)),
            'beam_size': str(kwargs.get('beam_size', 1))
        }
        
        # Add conditional parameters
        if kwargs.get('initial_prompt'):
            params['initial_prompt'] = kwargs['initial_prompt']
        
        if kwargs.get('suppress_tokens'):
            params['suppress_tokens'] = ','.join(map(str, kwargs['suppress_tokens']))
        
        return params
    
    def _parse_faster_whisper_response(self, result: Dict[str, Any], audio_file_path: str) -> STTResponse:
        """
        Parse Faster Whisper response into STTResponse format.
        
        Response Processing:
        - Extract main transcript
        - Parse word-level timestamps
        - Calculate confidence scores
        - Format metadata
        """
        
        try:
            # Extract main transcript
            transcript = result.get('text', '').strip()
            
            # Extract segments with timestamps
            segments = result.get('segments', [])
            word_timestamps = []
            speaker_segments = []
            
            overall_confidence = 0.0
            total_words = 0
            
            for segment in segments:
                # Segment-level information
                segment_start = segment.get('start', 0.0)
                segment_end = segment.get('end', 0.0)
                segment_text = segment.get('text', '').strip()
                
                speaker_segments.append(SpeakerSegment(
                    speaker_id="speaker_1",  # Faster Whisper doesn't do diarization
                    start_time=segment_start,
                    end_time=segment_end,
                    text=segment_text,
                    confidence=segment.get('avg_logprob', 0.0)
                ))
                
                # Word-level timestamps
                words = segment.get('words', [])
                for word_data in words:
                    word_timestamps.append(WordTimestamp(
                        word=word_data.get('word', '').strip(),
                        start_time=word_data.get('start', segment_start),
                        end_time=word_data.get('end', segment_end),
                        confidence=word_data.get('probability', 0.0)
                    ))
                    
                    # Accumulate confidence
                    overall_confidence += word_data.get('probability', 0.0)
                    total_words += 1
            
            # Calculate average confidence
            avg_confidence = overall_confidence / max(total_words, 1)
            
            # Detect language
            language_detected = result.get('language', 'unknown')
            
            # Calculate duration
            audio_duration = max([seg.get('end', 0.0) for seg in segments] + [0.0])
            
            return STTResponse(
                success=True,
                transcript=transcript,
                language_detected=language_detected,
                confidence=avg_confidence,
                word_timestamps=word_timestamps,
                speaker_segments=speaker_segments,
                duration_seconds=audio_duration,
                provider="faster_whisper",
                model_used=result.get('model', self.model),
                processing_metadata={
                    'segments_count': len(segments),
                    'words_count': total_words,
                    'audio_file': os.path.basename(audio_file_path),
                    'language_probability': result.get('language_probability', 0.0)
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to parse Faster Whisper response: {e}")
            return STTResponse(
                success=False,
                error_message=f"Response parsing failed: {str(e)}",
                provider="faster_whisper"
            )
```

#### Google Cloud Speech Client

```python
class GoogleSpeechClient(STTClient):
    """
    Google Cloud Speech-to-Text client with advanced features.
    
    Features:
    - High-quality transcription
    - Speaker diarization
    - Real-time processing
    - Multiple language support
    - Custom vocabulary
    - Punctuation and formatting
    """
    
    def __init__(self, config: STTConfig, credential_manager: Optional['CredentialManager'] = None):
        """
        Initialize Google Speech client.
        
        Initialization:
        - Credentials validation
        - Client setup
        - Feature configuration
        - Model selection
        """
        
        super().__init__(config)
        self.credential_manager = credential_manager or default_credential_manager
        
        # Initialize Google Speech client
        self._initialize_google_client()
        
        # Advanced features configuration
        self.enable_automatic_punctuation = config.enable_automatic_punctuation
        self.enable_speaker_diarization = config.enable_speaker_diarization
        self.max_speakers = config.max_speakers or 6
        self.use_enhanced_model = config.use_enhanced_model
        
        # Performance settings
        self.optimal_formats = ["wav", "flac", "ogg"]
        self.optimal_sample_rate = 16000
    
    def _initialize_google_client(self):
        """
        Initialize Google Cloud Speech client with proper authentication.
        
        Authentication Methods:
        - Service account key file
        - Application default credentials
        - Environment variables
        """
        
        try:
            from google.cloud import speech
            
            # Check for explicit credentials path
            credentials_path = self.credential_manager.get_credential("GOOGLE_APPLICATION_CREDENTIALS")
            
            if credentials_path and os.path.exists(credentials_path):
                # Use service account file
                self.client = speech.SpeechClient.from_service_account_json(credentials_path)
                logger.info("✓ Google Speech initialized with service account")
            else:
                # Use application default credentials
                self.client = speech.SpeechClient()
                logger.info("✓ Google Speech initialized with default credentials")
                
        except Exception as e:
            logger.error(f"Google Speech client initialization failed: {e}")
            raise RuntimeError(f"Google Speech initialization failed: {e}")
    
    def speech_to_text(self, audio_file_path: str, **kwargs) -> STTResponse:
        """
        Transcribe audio using Google Cloud Speech-to-Text.
        
        Transcription Features:
        - Long-form audio support
        - Speaker diarization
        - Custom vocabulary
        - Profanity filtering
        - Automatic punctuation
        """
        
        try:
            # Prepare audio data
            audio_data = self._prepare_audio_data(audio_file_path)
            
            # Build recognition config
            recognition_config = self._build_recognition_config(kwargs)
            
            # Choose processing method based on audio length
            if audio_data['duration'] > 60:  # Long audio
                response = self._transcribe_long_audio(audio_data, recognition_config)
            else:  # Short audio
                response = self._transcribe_short_audio(audio_data, recognition_config)
            
            # Parse Google Speech response
            return self._parse_google_response(response, audio_file_path)
            
        except Exception as e:
            logger.error(f"Google Speech transcription failed: {e}")
            return STTResponse(
                success=False,
                error_message=f"Google Speech transcription failed: {str(e)}",
                provider="google"
            )
    
    def _build_recognition_config(self, kwargs: Dict[str, Any]) -> 'speech.RecognitionConfig':
        """
        Build Google Speech recognition configuration.
        
        Configuration Options:
        - Language and model selection
        - Audio encoding settings
        - Advanced features
        - Quality enhancements
        """
        
        from google.cloud import speech
        
        # Base configuration
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=kwargs.get('sample_rate', self.optimal_sample_rate),
            language_code=kwargs.get('language', self.config.language or 'en-US'),
            enable_automatic_punctuation=kwargs.get('enable_automatic_punctuation', 
                                                   self.enable_automatic_punctuation),
            enable_word_time_offsets=kwargs.get('enable_word_timestamps', 
                                               self.config.enable_word_timestamps),
            enable_word_confidence=True,
            max_alternatives=kwargs.get('max_alternatives', 1),
            profanity_filter=kwargs.get('profanity_filter', False)
        )
        
        # Enhanced model selection
        if self.use_enhanced_model:
            config.use_enhanced = True
            config.model = kwargs.get('model', 'latest_long')
        
        # Speaker diarization configuration
        if kwargs.get('enable_speaker_diarization', self.enable_speaker_diarization):
            diarization_config = speech.SpeakerDiarizationConfig(
                enable_speaker_diarization=True,
                min_speaker_count=kwargs.get('min_speakers', 1),
                max_speaker_count=kwargs.get('max_speakers', self.max_speakers)
            )
            config.diarization_config = diarization_config
        
        # Custom vocabulary (if provided)
        custom_vocabulary = kwargs.get('custom_vocabulary')
        if custom_vocabulary:
            speech_contexts = [
                speech.SpeechContext(phrases=custom_vocabulary)
            ]
            config.speech_contexts = speech_contexts
        
        return config
    
    def _transcribe_long_audio(self, audio_data: Dict, config: 'speech.RecognitionConfig') -> Any:
        """
        Transcribe long audio using async operation.
        
        Long Audio Processing:
        - Upload to Google Cloud Storage (if needed)
        - Start long-running operation
        - Poll for completion
        - Retrieve results
        """
        
        from google.cloud import speech
        
        # For long audio, we need to use async operation
        if audio_data['size'] > 10 * 1024 * 1024:  # 10MB limit for sync
            # Would need to upload to GCS and use uri
            # For now, fallback to chunked processing
            return self._transcribe_chunked_audio(audio_data, config)
        else:
            # Use long_running_recognize for files under 10MB
            audio = speech.RecognitionAudio(content=audio_data['content'])
            operation = self.client.long_running_recognize(config=config, audio=audio)
            
            # Wait for operation to complete
            response = operation.result(timeout=300)  # 5 minute timeout
            return response
    
    def _transcribe_short_audio(self, audio_data: Dict, config: 'speech.RecognitionConfig') -> Any:
        """
        Transcribe short audio using synchronous API.
        
        Short Audio Processing:
        - Direct API call
        - Immediate results
        - Lower latency
        """
        
        from google.cloud import speech
        
        audio = speech.RecognitionAudio(content=audio_data['content'])
        response = self.client.recognize(config=config, audio=audio)
        return response
    
    def _parse_google_response(self, response: Any, audio_file_path: str) -> STTResponse:
        """
        Parse Google Speech API response.
        
        Response Processing:
        - Extract transcript alternatives
        - Parse word-level timestamps
        - Process speaker diarization
        - Calculate confidence scores
        """
        
        try:
            if not response.results:
                return STTResponse(
                    success=False,
                    error_message="No transcription results from Google Speech",
                    provider="google"
                )
            
            # Combine all results
            transcript_parts = []
            word_timestamps = []
            speaker_segments = []
            total_confidence = 0.0
            result_count = 0
            
            for result in response.results:
                if not result.alternatives:
                    continue
                
                # Get best alternative
                alternative = result.alternatives[0]
                transcript_parts.append(alternative.transcript)
                total_confidence += alternative.confidence
                result_count += 1
                
                # Process word-level information
                for word_info in alternative.words:
                    word_timestamps.append(WordTimestamp(
                        word=word_info.word,
                        start_time=word_info.start_time.total_seconds(),
                        end_time=word_info.end_time.total_seconds(),
                        confidence=getattr(word_info, 'confidence', alternative.confidence)
                    ))
                    
                    # Create speaker segments from word info
                    speaker_tag = getattr(word_info, 'speaker_tag', 1)
                    speaker_segments.append(SpeakerSegment(
                        speaker_id=f"speaker_{speaker_tag}",
                        start_time=word_info.start_time.total_seconds(),
                        end_time=word_info.end_time.total_seconds(),
                        text=word_info.word,
                        confidence=getattr(word_info, 'confidence', alternative.confidence)
                    ))
            
            # Merge speaker segments
            merged_speaker_segments = self._merge_speaker_segments(speaker_segments)
            
            # Calculate overall metrics
            full_transcript = ' '.join(transcript_parts)
            avg_confidence = total_confidence / max(result_count, 1)
            duration = max([w.end_time for w in word_timestamps] + [0.0])
            
            return STTResponse(
                success=True,
                transcript=full_transcript,
                language_detected=self.config.language or 'en-US',
                confidence=avg_confidence,
                word_timestamps=word_timestamps,
                speaker_segments=merged_speaker_segments,
                duration_seconds=duration,
                provider="google",
                model_used=getattr(self.config, 'model', 'default'),
                processing_metadata={
                    'results_count': len(response.results),
                    'words_count': len(word_timestamps),
                    'speakers_count': len(set(seg.speaker_id for seg in merged_speaker_segments)),
                    'audio_file': os.path.basename(audio_file_path)
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to parse Google Speech response: {e}")
            return STTResponse(
                success=False,
                error_message=f"Response parsing failed: {str(e)}",
                provider="google"
            )
```

## 🎯 Advanced Features and Processing

### Audio Quality Assessment

```python
class AudioQualityAssessor:
    """
    Comprehensive audio quality assessment for STT optimization.
    
    Assessment Features:
    - Signal-to-noise ratio analysis
    - Dynamic range measurement
    - Frequency spectrum analysis
    - Speech presence detection
    - Clipping and distortion detection
    """
    
    def __init__(self):
        self.assessment_cache = {}
        self.cache_enabled = True
    
    def assess_audio_quality(self, audio_file_path: str) -> AudioQualityAssessment:
        """
        Perform comprehensive audio quality assessment.
        
        Assessment Pipeline:
        1. Load and analyze audio data
        2. Calculate quality metrics
        3. Detect common issues
        4. Generate recommendations
        5. Score overall quality
        """
        
        # Check cache
        if self.cache_enabled and audio_file_path in self.assessment_cache:
            return self.assessment_cache[audio_file_path]
        
        try:
            # Load audio for analysis
            audio_data = self._load_audio_for_analysis(audio_file_path)
            
            # Perform quality assessments
            assessment = AudioQualityAssessment(
                overall_score=0.0,
                signal_to_noise_ratio=self._calculate_snr(audio_data),
                dynamic_range=self._calculate_dynamic_range(audio_data),
                clipping_detected=self._detect_clipping(audio_data),
                silence_ratio=self._calculate_silence_ratio(audio_data),
                frequency_response=self._analyze_frequency_response(audio_data),
                speech_presence=self._detect_speech_presence(audio_data),
                recommendations=[]
            )
            
            # Calculate overall score
            assessment.overall_score = self._calculate_overall_score(assessment)
            
            # Generate recommendations
            assessment.recommendations = self._generate_recommendations(assessment)
            
            # Cache result
            if self.cache_enabled:
                self.assessment_cache[audio_file_path] = assessment
            
            return assessment
            
        except Exception as e:
            logger.error(f"Audio quality assessment failed: {e}")
            return AudioQualityAssessment(
                overall_score=0.5,  # Default to medium quality
                error_message=str(e)
            )
    
    def _load_audio_for_analysis(self, audio_file_path: str) -> np.ndarray:
        """Load audio data for quality analysis."""
        
        try:
            import librosa
            
            # Load with librosa for detailed analysis
            audio_data, sample_rate = librosa.load(
                audio_file_path, 
                sr=None,  # Keep original sample rate
                mono=False  # Keep channels separate for analysis
            )
            
            return {
                'data': audio_data,
                'sample_rate': sample_rate,
                'duration': len(audio_data) / sample_rate
            }
            
        except ImportError:
            # Fallback to scipy if librosa not available
            from scipy.io import wavfile
            import numpy as np
            
            sample_rate, audio_data = wavfile.read(audio_file_path)
            
            # Normalize to [-1, 1] range
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            
            return {
                'data': audio_data,
                'sample_rate': sample_rate,
                'duration': len(audio_data) / sample_rate
            }
    
    def _calculate_snr(self, audio_data: Dict) -> float:
        """
        Calculate signal-to-noise ratio.
        
        SNR Calculation:
        - Estimate noise floor from quiet segments
        - Calculate signal power from active segments
        - Compute ratio in dB
        """
        
        import numpy as np
        
        data = audio_data['data']
        
        # Handle multi-channel audio
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)  # Convert to mono
        
        # Calculate RMS energy in sliding windows
        window_size = int(0.1 * audio_data['sample_rate'])  # 100ms windows
        step_size = window_size // 2
        
        energies = []
        for i in range(0, len(data) - window_size, step_size):
            window = data[i:i + window_size]
            energy = np.sqrt(np.mean(window ** 2))
            energies.append(energy)
        
        energies = np.array(energies)
        
        # Estimate noise floor (bottom 20% of energies)
        noise_threshold = np.percentile(energies, 20)
        
        # Estimate signal level (top 20% of energies)
        signal_level = np.percentile(energies, 80)
        
        # Calculate SNR in dB
        if noise_threshold > 0:
            snr_db = 20 * np.log10(signal_level / noise_threshold)
        else:
            snr_db = 100  # Very high SNR if no noise detected
        
        return float(snr_db)
    
    def _detect_clipping(self, audio_data: Dict) -> bool:
        """
        Detect audio clipping/distortion.
        
        Clipping Detection:
        - Check for samples at maximum amplitude
        - Analyze sudden amplitude changes
        - Count consecutive maximum samples
        """
        
        import numpy as np
        
        data = audio_data['data']
        
        # Handle multi-channel audio
        if len(data.shape) > 1:
            data = np.abs(data).max(axis=1)
        else:
            data = np.abs(data)
        
        # Define clipping threshold (95% of maximum)
        clipping_threshold = 0.95
        
        # Count samples above threshold
        clipped_samples = np.sum(data > clipping_threshold)
        clipping_ratio = clipped_samples / len(data)
        
        # Detect consecutive clipped samples (indicates hard clipping)
        consecutive_clipping = False
        consecutive_count = 0
        max_consecutive = 0
        
        for sample in data:
            if sample > clipping_threshold:
                consecutive_count += 1
                max_consecutive = max(max_consecutive, consecutive_count)
            else:
                consecutive_count = 0
        
        # Consider clipping detected if:
        # - More than 0.1% of samples are clipped, OR
        # - More than 10 consecutive samples are clipped
        return clipping_ratio > 0.001 or max_consecutive > 10
```

### Real-time Processing Pipeline

```python
class RealTimeSTTProcessor:
    """
    Real-time speech-to-text processing with streaming capabilities.
    
    Features:
    - Live audio stream processing
    - Incremental transcription
    - Voice activity detection
    - Buffer management
    - Low-latency processing
    """
    
    def __init__(self, stt_service: STTService, buffer_duration: float = 2.0):
        """
        Initialize real-time STT processor.
        
        Configuration:
        - STT service for transcription
        - Buffer management settings
        - Voice activity detection
        - Streaming parameters
        """
        
        self.stt_service = stt_service
        self.buffer_duration = buffer_duration
        self.sample_rate = 16000
        self.channels = 1
        
        # Audio buffer management
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()
        
        # Voice activity detection
        self.vad = VoiceActivityDetector()
        
        # Streaming state
        self.is_streaming = False
        self.stream_thread = None
        
        # Results callback
        self.results_callback = None
    
    def start_streaming(self, audio_source, results_callback):
        """
        Start real-time audio streaming and processing.
        
        Streaming Pipeline:
        1. Capture audio from source
        2. Buffer audio chunks
        3. Detect voice activity
        4. Process speech segments
        5. Return incremental results
        """
        
        self.results_callback = results_callback
        self.is_streaming = True
        
        # Start streaming thread
        self.stream_thread = threading.Thread(
            target=self._stream_processing_loop,
            args=(audio_source,)
        )
        self.stream_thread.start()
    
    def _stream_processing_loop(self, audio_source):
        """
        Main streaming processing loop.
        
        Processing Loop:
        - Continuous audio capture
        - Buffer management
        - Voice activity detection
        - Transcription triggering
        - Result delivery
        """
        
        import pyaudio
        import numpy as np
        
        # Configure audio stream
        chunk_size = int(self.sample_rate * 0.1)  # 100ms chunks
        
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=chunk_size
        )
        
        try:
            while self.is_streaming:
                # Read audio chunk
                audio_chunk = stream.read(chunk_size, exception_on_overflow=False)
                audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
                
                # Add to buffer
                with self.buffer_lock:
                    self.audio_buffer.extend(audio_array)
                    
                    # Maintain buffer size
                    max_buffer_size = int(self.sample_rate * self.buffer_duration)
                    if len(self.audio_buffer) > max_buffer_size:
                        overflow = len(self.audio_buffer) - max_buffer_size
                        self.audio_buffer = self.audio_buffer[overflow:]
                
                # Check for voice activity and process if needed
                self._check_and_process_speech()
                
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
    
    def _check_and_process_speech(self):
        """
        Check for voice activity and trigger processing.
        
        Voice Activity Processing:
        - Analyze current buffer for speech
        - Determine processing trigger
        - Extract speech segment
        - Queue for transcription
        """
        
        with self.buffer_lock:
            if len(self.audio_buffer) < self.sample_rate:  # Need at least 1 second
                return
            
            # Analyze recent audio for voice activity
            recent_audio = np.array(self.audio_buffer[-int(self.sample_rate * 0.5):])
            
            # Check voice activity
            has_speech = self.vad.detect_speech(recent_audio, self.sample_rate)
            
            if has_speech and len(self.audio_buffer) >= int(self.sample_rate * 1.0):
                # Extract segment for processing
                segment_audio = np.array(self.audio_buffer)
                
                # Process segment asynchronously
                threading.Thread(
                    target=self._process_audio_segment,
                    args=(segment_audio.copy(),)
                ).start()
                
                # Clear buffer after processing
                self.audio_buffer = []
    
    def _process_audio_segment(self, audio_segment: np.ndarray):
        """
        Process audio segment for transcription.
        
        Segment Processing:
        - Save audio to temporary file
        - Run STT processing
        - Parse results
        - Deliver to callback
        """
        
        import tempfile
        import wave
        
        try:
            # Create temporary audio file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                temp_path = tmp_file.name
                
                # Write audio data as WAV
                with wave.open(temp_path, 'wb') as wav_file:
                    wav_file.setnchannels(self.channels)
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(self.sample_rate)
                    wav_file.writeframes(audio_segment.tobytes())
            
            # Process with STT service
            response = self.stt_service.speech_to_text(temp_path)
            
            # Deliver results
            if self.results_callback and response.success:
                self.results_callback(response)
                
        except Exception as e:
            logger.error(f"Real-time STT processing failed: {e}")
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except:
                pass
```

This comprehensive documentation provides deep technical insight into the STT service architecture, provider implementations, advanced features, and real-time processing capabilities.