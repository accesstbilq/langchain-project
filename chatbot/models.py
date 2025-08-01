from django.db import models
from django.conf import settings

class ChatSession(models.Model):
    # Unique identifier for the session (could be based on browser session or token)
    session_key = models.CharField(max_length=255, unique=True)

    # Timestamp when the session was created
    created_at = models.DateTimeField(auto_now_add=True)

    # Timestamp when the session was last updated
    updated_at = models.DateTimeField(auto_now=True)

    # List of processed URLs for this session (optional)
    processed_urls = models.JSONField(default=list, blank=True)

    # Associated user (optional, can be null for anonymous users)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        null=True,
        blank=True
    )

    def __str__(self):
        return f"Session {self.session_key}"


class ProcessedURL(models.Model):
    # Choices to track the URL processing status
    STATUS_CHOICES = [
        ('processing', 'Processing'),
        ('success', 'Success'),
        ('failed', 'Failed'),
    ]

    # The original URL submitted for processing
    url = models.URLField()

    # The title extracted from the URL (if available)
    title = models.CharField(max_length=500, blank=True)

    # Status of processing (processing/success/failed)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES)

    # Details about processing result or errors (JSON object)
    processing_details = models.JSONField(default=dict, blank=True)

    # Timestamp of when the URL was processed
    created_at = models.DateTimeField(auto_now_add=True)

    # Reference to the related chat session
    session = models.ForeignKey(
        ChatSession,
        related_name='urls',
        on_delete=models.CASCADE
    )

    def __str__(self):
        return f"{self.url} - {self.status}"

class ChatMessage(models.Model):
    # Role indicates whether the message is from user or assistant
    ROLE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
    ]

    # Role of the message sender (user or assistant)
    role = models.CharField(max_length=20, choices=ROLE_CHOICES)

    # The actual message content
    content = models.TextField()

    # Any associated source links or metadata (optional)
    sources = models.JSONField(default=list, blank=True)

    # Timestamp of when the message was created
    timestamp = models.DateTimeField(auto_now_add=True)

    # Reference to the related chat session
    session = models.ForeignKey(
        ChatSession,
        related_name='messages',
        on_delete=models.CASCADE
    )

    # Default ordering by timestamp (oldest to newest)
    class Meta:
        ordering = ['timestamp']

    def __str__(self):
        return f"[{self.timestamp}] {self.role}: {self.content[:50]}"
