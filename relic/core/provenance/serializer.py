"""
Serialize provenance data to W3C PROV JSON-LD.
"""

from typing import Any, Dict, Iterable, List, Optional

from .models import ProvActivity, ProvAgent, ProvEntity


def _entity_payload(entity: ProvEntity) -> Dict[str, Any]:
    payload = {"prov:type": "prov:Entity", "prov:label": entity.label}
    payload.update(entity.attributes)
    return payload


def _activity_payload(activity: ProvActivity) -> Dict[str, Any]:
    payload = {"prov:type": "prov:Activity", "prov:label": activity.label}
    if activity.start_time:
        payload["prov:startTime"] = activity.start_time
    if activity.end_time:
        payload["prov:endTime"] = activity.end_time
    payload.update(activity.attributes)
    return payload


def _agent_payload(agent: ProvAgent) -> Dict[str, Any]:
    payload = {"prov:type": "prov:Agent", "prov:label": agent.label}
    payload.update(agent.attributes)
    return payload


def serialize_prov_jsonld(
    entities: Iterable[ProvEntity],
    activities: Iterable[ProvActivity],
    agents: Iterable[ProvAgent],
    relations: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    context: str = "https://www.w3.org/ns/prov",
) -> Dict[str, Any]:
    """
    Serialize provenance data into a JSON-LD dictionary.

    Parameters
    ----------
    entities : Iterable[ProvEntity]
        Entities to include.
    activities : Iterable[ProvActivity]
        Activities to include.
    agents : Iterable[ProvAgent]
        Agents to include.
    relations : dict, optional
        PROV relation lists keyed by relation name.
    context : str, optional
        JSON-LD context URL.

    Returns
    -------
    dict
        JSON-LD payload.
    """
    payload: Dict[str, Any] = {"@context": context}
    payload["entity"] = {entity.identifier: _entity_payload(entity) for entity in entities}
    payload["activity"] = {activity.identifier: _activity_payload(activity) for activity in activities}
    payload["agent"] = {agent.identifier: _agent_payload(agent) for agent in agents}
    if relations:
        for key, value in relations.items():
            payload[key] = list(value)
    return payload
