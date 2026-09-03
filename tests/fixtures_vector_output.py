"""
Copyright (C) 2026 Laurent G. Courty

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License
as published by the Free Software Foundation; either version 2.1
of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.
"""

from typing import NamedTuple

from itzi_core.data_containers import (
    DrainageLinkAttributes,
    DrainageLinkTopology,
    DrainageNetworkAttributes,
    DrainageNetworkTopology,
    DrainageNodeAttributes,
    DrainageNodeTopology,
)
from itzi_core.drainage import CouplingTypes

expected_node_coords = {
    "N1": (100.0, 200.0),
    "N2": (150.0, 180.0),
    "N3": (200.0, 160.0),
}

expected_vertices = {
    "L1": [(100.0, 200.0), (125.0, 190.0), (150.0, 180.0)],
    "L2": [(150.0, 180.0), (175.0, 170.0), (200.0, 160.0)],
}


class DummyDrainageNetwork(NamedTuple):
    topology: DrainageNetworkTopology
    attributes: DrainageNetworkAttributes


def create_dummy_drainage_network(with_coords=True):
    """Create a dummy DrainageNetwork object with 3 nodes and 2 links."""

    # Create 3 drainage nodes with correct types from drainage.py
    node1_attributes = DrainageNodeAttributes(
        node_id="N1",
        node_type="junction",  # Valid: junction, outfall, divider, storage
        coupling_type=CouplingTypes.ORIFICE,
        coupling_flow=0.5,
        inflow=1.0,
        outflow=0.8,
        lateral_inflow=0.2,
        losses=0.1,
        overflow=0.0,
        depth=1.5,
        head=101.5,
        crest_elevation=100.0,
        invert_elevation=98.0,
        initial_depth=5 / 9,
        full_depth=2.0,
        surcharge_depth=0.5,
        ponding_area=10.0,
        volume=15.0,
        full_volume=20.0,
    )

    node2_attributes = DrainageNodeAttributes(
        node_id="N2",
        node_type="junction",
        coupling_type=CouplingTypes.FREE_WEIR,
        coupling_flow=0.3,
        inflow=0.8,
        outflow=0.6,
        lateral_inflow=0.1,
        losses=0.05,
        overflow=0.0,
        depth=1.2,
        head=99.2,
        crest_elevation=98.0,
        invert_elevation=96.0,
        initial_depth=0.3,
        full_depth=1.8,
        surcharge_depth=0.4,
        ponding_area=8.0,
        volume=9.6,
        full_volume=14.4,
    )

    node3_attributes = DrainageNodeAttributes(
        node_id="N3",
        node_type="outfall",
        coupling_type=CouplingTypes.NOT_COUPLED,
        coupling_flow=0.0,
        inflow=0.6,
        outflow=0.6,
        lateral_inflow=0.0,
        losses=0.0,
        overflow=0.0,
        depth=0.8,
        head=96.8,
        crest_elevation=96.0,
        invert_elevation=94.0,
        initial_depth=0.2,
        full_depth=1.5,
        surcharge_depth=0.3,
        ponding_area=5.0,
        volume=4.0,
        full_volume=7.5,
    )

    # Create node topology objects
    if with_coords:
        node1 = DrainageNodeTopology(node_id="N1", coordinates=expected_node_coords["N1"])
        node2 = DrainageNodeTopology(node_id="N2", coordinates=expected_node_coords["N2"])
        node3 = DrainageNodeTopology(node_id="N3", coordinates=expected_node_coords["N3"])
    else:
        node1 = DrainageNodeTopology(node_id="N1", coordinates=None)
        node2 = DrainageNodeTopology(node_id="N2", coordinates=None)
        node3 = DrainageNodeTopology(node_id="N3", coordinates=None)

    # Create 2 drainage links with correct types from drainage.py
    link1_attributes = DrainageLinkAttributes(
        link_id="L1",
        link_type="conduit",  # Valid: conduit, pump, orifice, weir, outlet
        flow=0.8,
        depth=0.6,
        volume=12.0,
        inlet_offset=0.0,
        outlet_offset=0.0,
        froude=0.4,
    )

    link2_attributes = DrainageLinkAttributes(
        link_id="L2",
        link_type="conduit",
        flow=0.6,
        depth=0.5,
        volume=10.0,
        inlet_offset=0.0,
        outlet_offset=0.0,
        froude=0.3,
    )

    # Create link topology objects with vertices connecting the nodes
    if with_coords:
        link1 = DrainageLinkTopology(
            link_id="L1",
            start_node_id="N1",
            end_node_id="N2",
            vertices=expected_vertices["L1"],  # N1 to N2
        )

        link2 = DrainageLinkTopology(
            link_id="L2",
            start_node_id="N2",
            end_node_id="N3",
            vertices=expected_vertices["L2"],  # N2 to N3
        )
    else:
        link1 = DrainageLinkTopology(
            link_id="L1",
            start_node_id="N1",
            end_node_id="N2",
            vertices=None,
        )

        link2 = DrainageLinkTopology(
            link_id="L2",
            start_node_id="N2",
            end_node_id="N3",
            vertices=None,
        )

    return DummyDrainageNetwork(
        topology=DrainageNetworkTopology(nodes=(node1, node2, node3), links=(link1, link2)),
        attributes=DrainageNetworkAttributes(
            nodes=(node1_attributes, node2_attributes, node3_attributes),
            links=(link1_attributes, link2_attributes),
        ),
    )
