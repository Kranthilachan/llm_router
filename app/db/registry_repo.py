import json
from typing import Any, Dict, List

from app.db.connection import get_connection

def load_registry() -> List[Dict[str, Any]]:
    """Loads the entire model registry from SQLite and reconstructs the dictionary format."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # We join all model tables to get a complete picture
    query = """
    SELECT 
        m.id, m.name, m.provider, m.tier, m.status, m.api_available, m.open_weight, m.context_window,
        c.json_mode, c.function_calling, c.web_search, c.ocr, c.citation_support,
        c.modality_text, c.modality_image, c.modality_audio, c.modality_video, 
        c.modality_pdf, c.modality_spreadsheet, c.modality_presentation,
        metrics.performance_json, metrics.benchmarks_json, metrics.domains_json, 
        metrics.safety_json, metrics.verifier_fit_json,
        ops.input_cost_usd, ops.output_cost_usd, ops.relative_cost_score,
        ops.ops_static_json, ops.ops_dynamic_json,
        priors.global_alpha, priors.global_beta, priors.task_family_priors_json,
        priors.evaluation_stats_json, priors.routing_json
    FROM models m
    LEFT JOIN model_capabilities c ON m.id = c.model_id
    LEFT JOIN model_metrics metrics ON m.id = metrics.model_id
    LEFT JOIN model_ops ops ON m.id = ops.model_id
    LEFT JOIN model_priors priors ON m.id = priors.model_id
    """
    
    cursor.execute(query)
    rows = cursor.fetchall()
    
    registry = []
    for row in rows:
        entry = {
            "name": row["name"],
            "provider": row["provider"],
            "tier": row["tier"],
            "status": row["status"],
            "api_available": bool(row["api_available"]),
            "open_weight": bool(row["open_weight"]),
            "context": {"window": row["context_window"]},
            
            "capabilities": {
                "json_mode": bool(row["json_mode"]),
                "function_calling": bool(row["function_calling"]),
                "web_search": bool(row["web_search"]),
                "ocr": bool(row["ocr"]),
                "citation_support": bool(row["citation_support"])
            },
            "modalities": {
                "text": bool(row["modality_text"]),
                "image": bool(row["modality_image"]),
                "audio": bool(row["modality_audio"]),
                "video": bool(row["modality_video"]),
                "pdf": bool(row["modality_pdf"]),
                "spreadsheet": bool(row["modality_spreadsheet"]),
                "presentation": bool(row["modality_presentation"])
            },
            
            "performance": json.loads(row["performance_json"]) if row["performance_json"] else {},
            "benchmarks": json.loads(row["benchmarks_json"]) if row["benchmarks_json"] else {},
            "domains": json.loads(row["domains_json"]) if row["domains_json"] else {},
            "safety": json.loads(row["safety_json"]) if row["safety_json"] else {},
            "verifier_fit": json.loads(row["verifier_fit_json"]) if row["verifier_fit_json"] else {},
            
            "pricing": {
                "input_cost": row["input_cost_usd"],
                "output_cost": row["output_cost_usd"],
                "relative_cost_score": row["relative_cost_score"]
            },
            "ops_static": json.loads(row["ops_static_json"]) if row["ops_static_json"] else {},
            "ops_dynamic": json.loads(row["ops_dynamic_json"]) if row["ops_dynamic_json"] else {},
            
            "priors": {
                "global": {"alpha": row["global_alpha"], "beta": row["global_beta"]},
                "task_family": json.loads(row["task_family_priors_json"]) if row["task_family_priors_json"] else {}
            },
            "evaluation": json.loads(row["evaluation_stats_json"]) if row["evaluation_stats_json"] else {},
            "routing": json.loads(row["routing_json"]) if row["routing_json"] else {}
        }
        registry.append(entry)
        
    conn.close()
    return registry


def update_model_outcome(
    model_name: str,
    task_family: str,
    success: bool,
    quality_score: float = 0.0,
    latency_ms: float = 0.0,
    cost_usd: float = 0.0,
    user_accepted: bool = True,
    safety_flagged: bool = False,
):
    from datetime import datetime, timezone
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT id FROM models WHERE name = ?", (model_name,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        raise ValueError(f"Model {model_name} not found")
    model_id = row["id"]
    
    cursor.execute("SELECT evaluation_stats_json, task_family_priors_json FROM model_priors WHERE model_id = ?", (model_id,))
    priors_row = cursor.fetchone()
    if priors_row:
        ev = json.loads(priors_row["evaluation_stats_json"]) if priors_row["evaluation_stats_json"] else {}
        fam_priors = json.loads(priors_row["task_family_priors_json"]) if priors_row["task_family_priors_json"] else {}
        
        ev["samples"] = ev.get("samples", 0) + 1
        ev["quality_sum"] = ev.get("quality_sum", 0.0) + quality_score
        ev["latency_sum_ms"] = ev.get("latency_sum_ms", 0.0) + latency_ms
        ev["cost_sum"] = ev.get("cost_sum", 0.0) + cost_usd
        ev["last_updated"] = datetime.now(timezone.utc).isoformat()
        
        if success:
            ev["wins"] = ev.get("wins", 0) + 1
        else:
            ev["losses"] = ev.get("losses", 0) + 1
        
        if user_accepted:
            ev["user_acceptance_sum"] = ev.get("user_acceptance_sum", 0.0) + 1.0
        
        if safety_flagged:
            ev["safety_flags"] = ev.get("safety_flags", 0) + 1
            
        fam_prior = fam_priors.get(task_family)
        if fam_prior:
            if success:
                fam_prior["alpha"] = fam_prior.get("alpha", 1.0) + 1
            else:
                fam_prior["beta"] = fam_prior.get("beta", 1.0) + 1
                
        cursor.execute(
            "UPDATE model_priors SET evaluation_stats_json = ?, task_family_priors_json = ? WHERE model_id = ?",
            (json.dumps(ev), json.dumps(fam_priors), model_id)
        )
        conn.commit()
    conn.close()

def add_model(m: Dict[str, Any]) -> int:
    """Inserts a new model dictionary into the SQLite database."""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
        INSERT OR REPLACE INTO models (name, provider, tier, status, api_available, open_weight, context_window)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            m['name'], m.get('provider', 'Unknown'), m.get('tier', 'Mid'), m.get('status', 'active'), 
            bool(m.get('api_available', True)), bool(m.get('open_weight', False)), 
            m.get('context', {}).get('window', 8192)
        ))
        
        cursor.execute('SELECT id FROM models WHERE name = ?', (m['name'],))
        model_id = cursor.fetchone()[0]
        
        cap = m.get('capabilities', {})
        mod = m.get('modalities', {})
        cursor.execute('''
        INSERT OR REPLACE INTO model_capabilities (
            model_id, json_mode, function_calling, web_search, ocr, citation_support,
            modality_text, modality_image, modality_audio, modality_video, modality_pdf,
            modality_spreadsheet, modality_presentation
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            model_id,
            bool(cap.get('json_mode', False)), bool(cap.get('function_calling', False)),
            bool(cap.get('web_search', False)), bool(cap.get('ocr', False)), bool(cap.get('citation_support', False)),
            bool(mod.get('text', True)), bool(mod.get('image', False)), bool(mod.get('audio', False)),
            bool(mod.get('video', False)), bool(mod.get('pdf', False)), bool(mod.get('spreadsheet', False)),
            bool(mod.get('presentation', False))
        ))
        
        cursor.execute('''
        INSERT OR REPLACE INTO model_metrics (
            model_id, performance_json, benchmarks_json, domains_json, safety_json, verifier_fit_json
        ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            model_id,
            json.dumps(m.get('performance', {})),
            json.dumps(m.get('benchmarks', {})),
            json.dumps(m.get('domains', {})),
            json.dumps(m.get('safety', {})),
            json.dumps(m.get('verifier_fit', {}))
        ))
        
        pricing = m.get('pricing', {})
        cursor.execute('''
        INSERT OR REPLACE INTO model_ops (
            model_id, input_cost_usd, output_cost_usd, relative_cost_score, ops_static_json, ops_dynamic_json
        ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            model_id,
            pricing.get('input_cost', 0.0), pricing.get('output_cost', 0.0), pricing.get('relative_cost_score', 0.0),
            json.dumps(m.get('ops_static', {})),
            json.dumps(m.get('ops_dynamic', {}))
        ))
        
        priors = m.get('priors', {})
        global_priors = priors.get('global', {})
        cursor.execute('''
        INSERT OR REPLACE INTO model_priors (
            model_id, global_alpha, global_beta, task_family_priors_json, evaluation_stats_json, routing_json
        ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            model_id,
            global_priors.get('alpha', 1.0), global_priors.get('beta', 1.0),
            json.dumps(priors.get('task_family', {})),
            json.dumps(m.get('evaluation', {"samples": 0, "wins": 0, "losses": 0, "quality_sum": 0.0, "latency_sum_ms": 0.0, "cost_sum": 0.0, "user_acceptance_sum": 0.0, "safety_flags": 0})),
            json.dumps(m.get('routing', {}))
        ))
        
        conn.commit()
        return model_id
    finally:
        conn.close()
