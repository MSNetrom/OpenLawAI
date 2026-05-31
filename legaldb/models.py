from __future__ import annotations

from django.db import models


class TimeStampedModel(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class LegalSource(TimeStampedModel):
    code = models.CharField(max_length=16, primary_key=True)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    requires_authentication = models.BooleanField(default=False)

    class Meta:
        verbose_name = "Legal source"
        verbose_name_plural = "Legal sources"

    def __str__(self) -> str:
        return f"{self.code}: {self.name}"


class LegalArea(TimeStampedModel):
    code = models.CharField(max_length=32, unique=True)
    title = models.CharField(max_length=255)
    parent = models.ForeignKey(
        "self", related_name="children", null=True, blank=True, on_delete=models.SET_NULL
    )

    class Meta:
        verbose_name = "Legal area"
        verbose_name_plural = "Legal areas"
        ordering = ["code"]

    def __str__(self) -> str:
        return self.title


class Organization(TimeStampedModel):
    class OrganizationType(models.TextChoices):
        MINISTRY = "ministry", "Ministry"
        SUBUNIT = "subunit", "Subunit"
        AGENCY = "agency", "Agency"
        OTHER = "other", "Other"

    name = models.CharField(max_length=255)
    type = models.CharField(max_length=16, choices=OrganizationType.choices, default=OrganizationType.OTHER)
    parent = models.ForeignKey(
        "self", related_name="suborganizations", null=True, blank=True, on_delete=models.SET_NULL
    )

    class Meta:
        verbose_name = "Organization"
        verbose_name_plural = "Organizations"
        unique_together = ("name", "type", "parent")

    def __str__(self) -> str:
        return self.name


class DocumentWork(TimeStampedModel):
    class DocumentType(models.TextChoices):
        LAW = "law", "Law"
        FORSKRIFT = "forskrift", "Forskrift"
        OTHER = "other", "Other"

    class DocumentStatus(models.TextChoices):
        ACTIVE = "active", "Active"
        HISTORIC = "historic", "Historic"
        REVOKED = "revoked", "Revoked"

    ref_id = models.CharField(max_length=255, unique=True)
    legacy_id = models.CharField(max_length=64, blank=True)
    legal_source = models.ForeignKey(
        LegalSource, related_name="works", on_delete=models.PROTECT, null=True, blank=True
    )
    document_type = models.CharField(
        max_length=16, choices=DocumentType.choices, default=DocumentType.LAW, db_index=True
    )
    status = models.CharField(max_length=16, choices=DocumentStatus.choices, default=DocumentStatus.ACTIVE, db_index=True)
    title = models.CharField(max_length=512, blank=True)
    short_title = models.CharField(max_length=255, blank=True)
    language = models.CharField(max_length=32, blank=True)
    applies_to = models.JSONField(default=list, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    date_in_force = models.DateField(null=True, blank=True)
    date_of_publication = models.DateTimeField(null=True, blank=True)
    misc_information = models.TextField(blank=True)

    legal_areas = models.ManyToManyField(LegalArea, related_name="works", blank=True)
    organizations = models.ManyToManyField(
        Organization, through="DocumentOrganizationRole", related_name="works", blank=True
    )

    class Meta:
        verbose_name = "Document work"
        verbose_name_plural = "Document works"
        ordering = ["ref_id"]

    def __str__(self) -> str:
        return self.ref_id


class DocumentOrganizationRole(TimeStampedModel):
    class RoleType(models.TextChoices):
        MINISTRY = "ministry", "Ministry"
        SUBUNIT = "subunit", "Subunit"
        OWNER = "owner", "Owner"
        PUBLISHER = "publisher", "Publisher"

    work = models.ForeignKey(DocumentWork, related_name="organization_roles", on_delete=models.CASCADE)
    organization = models.ForeignKey(Organization, on_delete=models.CASCADE)
    role = models.CharField(max_length=16, choices=RoleType.choices)

    class Meta:
        verbose_name = "Document organization role"
        verbose_name_plural = "Document organization roles"
        unique_together = ("work", "organization", "role")

    def __str__(self) -> str:
        return f"{self.work.ref_id} - {self.organization.name} ({self.role})"


class DocumentVersion(TimeStampedModel):
    work = models.ForeignKey(DocumentWork, related_name="versions", on_delete=models.CASCADE)
    dok_id = models.CharField(max_length=255, unique=True)
    version_label = models.TextField(blank=True)
    source_path = models.CharField(max_length=512, blank=True)
    source_url = models.URLField(blank=True)
    checksum = models.CharField(max_length=128, blank=True)
    is_current = models.BooleanField(default=True, db_index=True)
    in_force = models.BooleanField(default=True, db_index=True)
    last_changed_at = models.DateTimeField(null=True, blank=True)
    last_change_in_force = models.DateTimeField(null=True, blank=True)
    last_changed_by = models.TextField(blank=True)
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        verbose_name = "Document version"
        verbose_name_plural = "Document versions"
        ordering = ["work", "-created_at"]
        constraints = [
            models.UniqueConstraint(
                fields=["work"],
                condition=models.Q(is_current=True),
                name="legaldb_single_current_version_per_work",
            ),
        ]

    def __str__(self) -> str:
        return self.dok_id


class DocumentSection(TimeStampedModel):
    version = models.ForeignKey(DocumentVersion, related_name="sections", on_delete=models.CASCADE)
    section_id = models.CharField(max_length=255)
    ref_id = models.CharField(max_length=255, blank=True)
    heading = models.CharField(max_length=512, blank=True)
    html = models.TextField()
    text = models.TextField()
    level = models.PositiveIntegerField(default=0)
    order = models.PositiveIntegerField()
    parent = models.ForeignKey(
        "self", related_name="children", null=True, blank=True, on_delete=models.CASCADE
    )

    class Meta:
        verbose_name = "Document section"
        verbose_name_plural = "Document sections"
        unique_together = ("version", "section_id")
        ordering = ["version", "order"]

    def __str__(self) -> str:
        return f"{self.version.dok_id}:{self.section_id}"


class DocumentRelationship(TimeStampedModel):
    class RelationType(models.TextChoices):
        BASED_ON = "based_on", "Based on"
        CHANGES = "changes", "Changes"
        REPEALS = "repeals", "Repeals"
        RELATED = "related", "Related"

    from_work = models.ForeignKey(
        DocumentWork, related_name="outgoing_relationships", on_delete=models.CASCADE
    )
    to_work = models.ForeignKey(
        DocumentWork, related_name="incoming_relationships", on_delete=models.CASCADE
    )
    relation_type = models.CharField(max_length=16, choices=RelationType.choices)
    evidence = models.CharField(max_length=255, blank=True)
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        verbose_name = "Document relationship"
        verbose_name_plural = "Document relationships"
        unique_together = ("from_work", "to_work", "relation_type")

    def __str__(self) -> str:
        return f"{self.from_work.ref_id} -> {self.to_work.ref_id} ({self.relation_type})"


class ChunkRef(TimeStampedModel):
    version = models.ForeignKey(DocumentVersion, related_name="chunks", on_delete=models.CASCADE)
    section = models.ForeignKey(
        DocumentSection, related_name="chunks", null=True, blank=True, on_delete=models.SET_NULL
    )
    chunk_id = models.CharField(max_length=255)
    vector_store_id = models.CharField(max_length=64, db_index=True)
    order = models.PositiveIntegerField()
    char_start = models.PositiveIntegerField(null=True, blank=True)
    char_end = models.PositiveIntegerField(null=True, blank=True)
    token_count = models.PositiveIntegerField(null=True, blank=True)
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        verbose_name = "Chunk reference"
        verbose_name_plural = "Chunk references"
        unique_together = ("version", "chunk_id")
        ordering = ["version", "order"]

    def __str__(self) -> str:
        return f"{self.version.dok_id}::{self.chunk_id}"
