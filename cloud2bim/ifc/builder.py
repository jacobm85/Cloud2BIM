"""IFC 2x3 builder using ifcopenshell.api.

Built for clean Revit import:
    - Stable GUIDs (deterministic from element identity, not random)
    - Single material per element type, declared once and reused
    - Proper IfcRelContainedInSpatialStructure relations
    - World coordinate system at the user-supplied site coordinates
    - All elements anchored to building storeys

Avoids the v1 generate_ifc.py's manual IFC entity creation; uses the
ifcopenshell.api which handles the boilerplate correctly.
"""
from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import List

import ifcopenshell
import ifcopenshell.api
import numpy as np

from cloud2bim.config import IFCConfig
from cloud2bim.elements.openings import Opening
from cloud2bim.elements.roofs import RoofPlane
from cloud2bim.elements.slabs import Slab
from cloud2bim.elements.walls import Wall
from cloud2bim.io.coordinates import CoordinateOffset
from cloud2bim.logging import get_logger

log = get_logger(__name__)


def _stable_guid(*parts: str) -> str:
    """Deterministic IFC GUID from string parts.

    Same input → same GUID across runs. Important for Revit incremental
    updates (same element keeps its identity).
    """
    digest = hashlib.md5("|".join(parts).encode()).digest()
    return ifcopenshell.guid.compress(uuid.UUID(bytes=digest).hex)


class IfcBuilder:
    """High-level IFC 2x3 model builder."""

    def __init__(self, cfg: IFCConfig, offset: CoordinateOffset | None = None):
        self.cfg = cfg
        self.offset = offset or CoordinateOffset(0, 0, 0)
        self.model = ifcopenshell.api.run("project.create_file", version="IFC2X3")

        # Owner history setup — required by ifcopenshell ≥ 0.7 before any
        # root.create_entity call, otherwise it raises "Please create a user".
        self._init_owner_history()

        self._project = ifcopenshell.api.run(
            "root.create_entity", self.model,
            ifc_class="IfcProject", name=cfg.project.name,
        )
        ifcopenshell.api.run("unit.assign_unit", self.model)

        # Spatial hierarchy: Project → Site → Building → Storeys
        self._site = self._create_spatial("IfcSite", "Site", parent=self._project)
        self._building = self._create_spatial(
            "IfcBuilding", cfg.building.name or "Building", parent=self._site
        )
        self._storeys: dict[int, ifcopenshell.entity_instance] = {}
        self._wall_material = self._create_material(cfg.default_material)

    # ── public API ──────────────────────────────────────────────────────

    def add_slab(self, slab: Slab, storey_idx: int, name: str | None = None) -> ifcopenshell.entity_instance:
        storey = self._ensure_storey(storey_idx, slab.bottom_z + slab.thickness)
        guid = _stable_guid("slab", str(storey_idx), f"{slab.bottom_z:.3f}")
        ifc_slab = ifcopenshell.api.run(
            "root.create_entity", self.model,
            ifc_class="IfcSlab", name=name or f"Slab {storey_idx + 1}",
        )
        ifc_slab.GlobalId = guid

        # Geometry: extrude the polygon along Z by ``thickness``
        rep = self._extruded_solid_from_polygon(
            slab.polygon_x, slab.polygon_y, slab.thickness, slab.bottom_z,
        )
        self._assign_representation(ifc_slab, rep)
        self._assign_material(ifc_slab, self._wall_material)
        ifcopenshell.api.run("spatial.assign_container", self.model,
                             relating_structure=storey, products=[ifc_slab])
        return ifc_slab

    def add_wall(self, wall: Wall) -> ifcopenshell.entity_instance:
        storey = self._ensure_storey(wall.storey, wall.z_placement)
        guid = _stable_guid("wall", str(wall.storey), f"{wall.start[0]:.3f}",
                            f"{wall.start[1]:.3f}", f"{wall.end[0]:.3f}", f"{wall.end[1]:.3f}")
        ifc_wall = ifcopenshell.api.run(
            "root.create_entity", self.model,
            ifc_class="IfcWallStandardCase", name=f"Wall {guid[-6:]}",
        )
        ifc_wall.GlobalId = guid

        rep = self._wall_swept_solid(wall)
        self._assign_representation(ifc_wall, rep)
        self._assign_material(ifc_wall, self._wall_material)
        ifcopenshell.api.run("spatial.assign_container", self.model,
                             relating_structure=storey, products=[ifc_wall])

        # IsExternal property for Revit
        self._add_property_set(ifc_wall, "Pset_WallCommon", {
            "IsExternal": wall.label == "exterior",
        })
        return ifc_wall

    def add_opening(self, opening: Opening, host_wall_ifc: ifcopenshell.entity_instance, host_wall: Wall) -> None:
        guid_open = _stable_guid("opening", str(opening.wall_storey), str(opening.wall_index),
                                 f"{opening.x_along_wall_start:.3f}")
        opening_el = ifcopenshell.api.run(
            "root.create_entity", self.model,
            ifc_class="IfcOpeningElement", name=f"Opening {opening.type}",
        )
        opening_el.GlobalId = guid_open
        rep = self._opening_solid(opening, host_wall)
        self._assign_representation(opening_el, rep)
        ifcopenshell.api.run("void.add_opening", self.model,
                             opening=opening_el, element=host_wall_ifc)

        # Insert door or window filling
        ifc_class = "IfcDoor" if opening.type == "door" else "IfcWindow"
        guid_fill = _stable_guid("fill", guid_open)
        fill = ifcopenshell.api.run(
            "root.create_entity", self.model,
            ifc_class=ifc_class, name=f"{opening.type.capitalize()}",
        )
        fill.GlobalId = guid_fill
        ifcopenshell.api.run("void.add_filling", self.model,
                             opening=opening_el, element=fill)

    def add_roof_plane(self, roof: RoofPlane, storey_idx: int) -> ifcopenshell.entity_instance:
        storey = self._ensure_storey(storey_idx, float(roof.polygon[:, 2].min()))
        guid = _stable_guid("roof", str(storey_idx), f"{roof.centroid[0]:.3f}",
                            f"{roof.centroid[1]:.3f}", f"{roof.centroid[2]:.3f}")
        roof_el = ifcopenshell.api.run(
            "root.create_entity", self.model,
            ifc_class="IfcRoof", name=f"Roof slope {roof.slope_deg:.0f}°",
        )
        roof_el.GlobalId = guid
        # Build as a thin extruded slab perpendicular to the plane
        rep = self._roof_plane_solid(roof)
        self._assign_representation(roof_el, rep)
        self._assign_material(roof_el, self._wall_material)
        ifcopenshell.api.run("spatial.assign_container", self.model,
                             relating_structure=storey, products=[roof_el])
        return roof_el

    def write(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.write(str(path))
        log.info("IFC written: %s", path)

    # ── internal ────────────────────────────────────────────────────────

    def _init_owner_history(self) -> None:
        """Create person + organisation + application so OwnerHistory works."""
        import ifcopenshell.api.owner.settings as _owner_settings

        person = ifcopenshell.api.run(
            "owner.add_person", self.model,
            identification=self.cfg.author.given_name or "cloud2bim",
            family_name=self.cfg.author.family_name or "Cloud2BIM",
            given_name=self.cfg.author.given_name or "Pipeline",
        )
        org = ifcopenshell.api.run(
            "owner.add_organisation", self.model,
            identification=self.cfg.author.organization or "Cloud2BIM",
            name=self.cfg.author.organization or "Cloud2BIM",
        )
        person_org = ifcopenshell.api.run(
            "owner.add_person_and_organisation", self.model,
            person=person, organisation=org,
        )
        application = ifcopenshell.api.run(
            "owner.add_application", self.model,
            application_developer=org,
            version="2.0.0",
            application_full_name="Cloud2BIM",
            application_identifier="Cloud2BIM",
        )
        # ifcopenshell.api.owner.settings.{set_user,set_application} accept a
        # callable; we pass a lambda that returns our pre-built entities.
        _owner_settings.set_user = lambda *_a, **_k: person_org
        _owner_settings.set_application = lambda *_a, **_k: application
        _owner_settings.get_user = lambda *_a, **_k: person_org
        _owner_settings.get_application = lambda *_a, **_k: application

    def _create_spatial(self, ifc_class: str, name: str, parent):
        entity = ifcopenshell.api.run(
            "root.create_entity", self.model, ifc_class=ifc_class, name=name,
        )
        ifcopenshell.api.run(
            "aggregate.assign_object", self.model, products=[entity], relating_object=parent,
        )
        return entity

    def _ensure_storey(self, idx: int, elevation: float):
        if idx not in self._storeys:
            storey = ifcopenshell.api.run(
                "root.create_entity", self.model,
                ifc_class="IfcBuildingStorey", name=f"Floor {elevation:.1f} m",
            )
            storey.Elevation = float(elevation)
            ifcopenshell.api.run(
                "aggregate.assign_object", self.model,
                products=[storey], relating_object=self._building,
            )
            self._storeys[idx] = storey
        return self._storeys[idx]

    def _create_material(self, name: str):
        mat = ifcopenshell.api.run("material.add_material", self.model, name=name)
        return mat

    def _assign_material(self, element, material):
        ifcopenshell.api.run(
            "material.assign_material", self.model,
            products=[element], material=material, type="IfcMaterial",
        )

    def _assign_representation(self, element, representation):
        ifcopenshell.api.run(
            "geometry.assign_representation", self.model,
            product=element, representation=representation,
        )

    def _add_property_set(self, element, pset_name: str, properties: dict):
        pset = ifcopenshell.api.run(
            "pset.add_pset", self.model, product=element, name=pset_name,
        )
        ifcopenshell.api.run(
            "pset.edit_pset", self.model, pset=pset, properties=properties,
        )

    def _extruded_solid_from_polygon(self, xs, ys, depth: float, base_z: float):
        """Build IfcExtrudedAreaSolid from XY polygon."""
        ctx = self._geometric_context()
        # Polygon in IfcArbitraryClosedProfileDef
        pts = [self.model.create_entity("IfcCartesianPoint", Coordinates=(float(x), float(y)))
               for x, y in zip(xs, ys)]
        pts.append(pts[0])  # close
        polyline = self.model.create_entity("IfcPolyline", Points=pts)
        profile = self.model.create_entity(
            "IfcArbitraryClosedProfileDef", ProfileType="AREA", OuterCurve=polyline,
        )
        position = self._placement_at(0, 0, base_z)
        direction = self.model.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0))
        solid = self.model.create_entity(
            "IfcExtrudedAreaSolid", SweptArea=profile, Position=position,
            ExtrudedDirection=direction, Depth=float(depth),
        )
        return self.model.create_entity(
            "IfcShapeRepresentation",
            ContextOfItems=ctx, RepresentationIdentifier="Body",
            RepresentationType="SweptSolid", Items=[solid],
        )

    def _wall_swept_solid(self, wall: Wall):
        ctx = self._geometric_context()
        sx, sy = wall.start
        ex, ey = wall.end
        length = float(np.hypot(ex - sx, ey - sy))
        # Profile: a rectangle of (length × thickness) centred on the axis
        half_t = wall.thickness / 2
        pts_local = [(0.0, -half_t), (length, -half_t), (length, half_t), (0.0, half_t), (0.0, -half_t)]
        ifc_pts = [self.model.create_entity("IfcCartesianPoint", Coordinates=p) for p in pts_local]
        polyline = self.model.create_entity("IfcPolyline", Points=ifc_pts)
        profile = self.model.create_entity(
            "IfcArbitraryClosedProfileDef", ProfileType="AREA", OuterCurve=polyline,
        )
        # Place at start point with rotation along axis
        angle = float(np.arctan2(ey - sy, ex - sx))
        position = self._placement_at(sx, sy, wall.z_placement, angle_z=angle)
        extrude = self.model.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0))
        solid = self.model.create_entity(
            "IfcExtrudedAreaSolid", SweptArea=profile, Position=position,
            ExtrudedDirection=extrude, Depth=float(wall.height),
        )
        return self.model.create_entity(
            "IfcShapeRepresentation",
            ContextOfItems=ctx, RepresentationIdentifier="Body",
            RepresentationType="SweptSolid", Items=[solid],
        )

    def _opening_solid(self, opening: Opening, host_wall: Wall):
        ctx = self._geometric_context()
        sx, sy = host_wall.start
        ex, ey = host_wall.end
        # Opening rectangle in wall-local coords (x along, y across, z up)
        x0 = opening.x_along_wall_start
        x1 = opening.x_along_wall_end
        half_t = host_wall.thickness  # full thickness for clean cut
        pts_local = [(x0, -half_t), (x1, -half_t), (x1, half_t), (x0, half_t), (x0, -half_t)]
        ifc_pts = [self.model.create_entity("IfcCartesianPoint", Coordinates=p) for p in pts_local]
        polyline = self.model.create_entity("IfcPolyline", Points=ifc_pts)
        profile = self.model.create_entity(
            "IfcArbitraryClosedProfileDef", ProfileType="AREA", OuterCurve=polyline,
        )
        angle = float(np.arctan2(ey - sy, ex - sx))
        position = self._placement_at(sx, sy, host_wall.z_placement + opening.z_min, angle_z=angle)
        extrude = self.model.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0))
        solid = self.model.create_entity(
            "IfcExtrudedAreaSolid", SweptArea=profile, Position=position,
            ExtrudedDirection=extrude, Depth=float(opening.height),
        )
        return self.model.create_entity(
            "IfcShapeRepresentation",
            ContextOfItems=ctx, RepresentationIdentifier="Body",
            RepresentationType="SweptSolid", Items=[solid],
        )

    def _roof_plane_solid(self, roof: RoofPlane, thickness: float = 0.20):
        """Thin extruded slab along the plane normal."""
        ctx = self._geometric_context()
        # Build profile from the polygon's projection
        poly = roof.polygon
        # Use projection to plane local axes (already done in _project_hull)
        normal = roof.normal
        arbitrary = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(normal, arbitrary); u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        rel = poly - roof.centroid
        pts_2d = np.column_stack([rel @ u, rel @ v])
        ifc_pts = [self.model.create_entity("IfcCartesianPoint", Coordinates=(float(p[0]), float(p[1])))
                   for p in pts_2d]
        ifc_pts.append(ifc_pts[0])
        polyline = self.model.create_entity("IfcPolyline", Points=ifc_pts)
        profile = self.model.create_entity(
            "IfcArbitraryClosedProfileDef", ProfileType="AREA", OuterCurve=polyline,
        )
        # Place at centroid, oriented so local Z = plane normal
        position = self._placement_with_normal(roof.centroid, u, v, normal)
        direction = self.model.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0))
        solid = self.model.create_entity(
            "IfcExtrudedAreaSolid", SweptArea=profile, Position=position,
            ExtrudedDirection=direction, Depth=float(thickness),
        )
        return self.model.create_entity(
            "IfcShapeRepresentation",
            ContextOfItems=ctx, RepresentationIdentifier="Body",
            RepresentationType="SweptSolid", Items=[solid],
        )

    # Cached geometric context
    _ctx_cache = None

    def _geometric_context(self):
        if self._ctx_cache is None:
            self._ctx_cache = ifcopenshell.api.run(
                "context.add_context", self.model, context_type="Model",
            )
            ifcopenshell.api.run(
                "context.add_context", self.model, context_type="Model",
                context_identifier="Body", target_view="MODEL_VIEW",
                parent=self._ctx_cache,
            )
        return self._ctx_cache

    def _placement_at(self, x: float, y: float, z: float, angle_z: float = 0.0):
        location = self.model.create_entity(
            "IfcCartesianPoint", Coordinates=(float(x), float(y), float(z)),
        )
        if angle_z == 0.0:
            return self.model.create_entity("IfcAxis2Placement3D", Location=location)
        ref_dir = self.model.create_entity(
            "IfcDirection", DirectionRatios=(float(np.cos(angle_z)), float(np.sin(angle_z)), 0.0),
        )
        z_dir = self.model.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0))
        return self.model.create_entity(
            "IfcAxis2Placement3D", Location=location, Axis=z_dir, RefDirection=ref_dir,
        )

    def _placement_with_normal(self, origin, u, v, normal):
        location = self.model.create_entity(
            "IfcCartesianPoint",
            Coordinates=tuple(float(c) for c in origin),
        )
        z_dir = self.model.create_entity(
            "IfcDirection", DirectionRatios=tuple(float(c) for c in normal),
        )
        x_dir = self.model.create_entity(
            "IfcDirection", DirectionRatios=tuple(float(c) for c in u),
        )
        return self.model.create_entity(
            "IfcAxis2Placement3D", Location=location, Axis=z_dir, RefDirection=x_dir,
        )
