from django.contrib import admin

from legaldb.models import (
    ChunkRef,
    DocumentOrganizationRole,
    DocumentRelationship,
    DocumentSection,
    DocumentVersion,
    DocumentWork,
    LegalArea,
    LegalSource,
    Organization,
)


@admin.register(LegalSource)
class LegalSourceAdmin(admin.ModelAdmin):
    list_display = ("code", "name", "requires_authentication", "updated_at")
    search_fields = ("code", "name")


@admin.register(LegalArea)
class LegalAreaAdmin(admin.ModelAdmin):
    list_display = ("code", "title", "parent", "updated_at")
    search_fields = ("code", "title")


@admin.register(Organization)
class OrganizationAdmin(admin.ModelAdmin):
    list_display = ("name", "type", "parent", "updated_at")
    list_filter = ("type",)
    search_fields = ("name",)


@admin.register(DocumentWork)
class DocumentWorkAdmin(admin.ModelAdmin):
    list_display = ("ref_id", "document_type", "status", "title", "updated_at")
    list_filter = ("document_type", "status")
    search_fields = ("ref_id", "title", "short_title")


@admin.register(DocumentOrganizationRole)
class DocumentOrganizationRoleAdmin(admin.ModelAdmin):
    list_display = ("work", "organization", "role", "updated_at")
    list_filter = ("role",)
    search_fields = ("work__ref_id", "organization__name")


@admin.register(DocumentVersion)
class DocumentVersionAdmin(admin.ModelAdmin):
    list_display = ("dok_id", "work", "is_current", "in_force", "updated_at")
    list_filter = ("is_current", "in_force")
    search_fields = ("dok_id", "work__ref_id")


@admin.register(DocumentSection)
class DocumentSectionAdmin(admin.ModelAdmin):
    list_display = ("version", "section_id", "heading", "order", "updated_at")
    search_fields = ("version__dok_id", "section_id", "heading", "ref_id")


@admin.register(DocumentRelationship)
class DocumentRelationshipAdmin(admin.ModelAdmin):
    list_display = ("from_work", "to_work", "relation_type", "updated_at")
    list_filter = ("relation_type",)
    search_fields = ("from_work__ref_id", "to_work__ref_id")


@admin.register(ChunkRef)
class ChunkRefAdmin(admin.ModelAdmin):
    list_display = ("version", "chunk_id", "vector_store_id", "order", "updated_at")
    search_fields = ("version__dok_id", "chunk_id", "vector_store_id")
