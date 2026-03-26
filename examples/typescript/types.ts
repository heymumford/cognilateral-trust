/**
 * TypeScript types for the cognilateral-trust REST API.
 *
 * Use with the hosted API at cognilateral.com or self-hosted.
 *
 * Example:
 *   const resp = await fetch(`${COGNILATERAL_API_BASE_URL}/api/trust/summary`);
 *   const data: TrustSummary = await resp.json();
 */

/** Default base URL for the hosted Cognilateral API. Override for self-hosted or staging. */
export const COGNILATERAL_API_BASE_URL = 'https://cognilateral.com';

export interface TrustEvaluation {
  confidence: number;
  tier: ConfidenceTier;
  route: 'basic' | 'warrant_check' | 'sovereignty_gate';
  should_proceed: boolean;
  verdict: 'ACT' | 'ESCALATE';
  requires_warrant_check: boolean;
  requires_sovereignty_gate: boolean;
  record_id: string | null;
  reasons: string[];
}

export type ConfidenceTier =
  'C0' | 'C1' | 'C2' | 'C3' | 'C4' | 'C5' | 'C6' | 'C7' | 'C8' | 'C9';

export interface AccountabilityRecord {
  record_id: string;
  timestamp: number;
  verdict: 'ACT' | 'ESCALATE';
  reasons: string[];
  context: Record<string, unknown>;
  confidence: number;
  confidence_tier: number | null;
}

export interface TrustSummary {
  calibration: {
    accuracy: number;
    accuracy_note: string;
    weights: Record<string, number>;
    predictions_total?: number;
    predictions_resolved?: number;
    predictions_pending?: number;
  };
  warrants: {
    available: boolean;
    total?: number;
    active?: number;
    tier_distribution?: Record<string, number>;
  };
  sovereignty: {
    recent_decisions: number;
    act_count: number;
    escalate_count: number;
    act_rate: number;
  };
  tier_routing: Record<ConfidenceTier, string>;
}

export interface SovereigntyRequest {
  confidence?: number;
  confidence_tier?: number;
  is_reversible?: boolean;
  tests_pass?: boolean;
  module_count?: number;
  welfare_affected?: boolean;
  touches_external?: boolean;
}

export interface SovereigntyResponse {
  verdict: 'ACT' | 'ESCALATE';
  rationale: string;
  passed: boolean;
  confidence_tier: number | null;
  effective_confidence: number;
}

/**
 * Minimal client for the cognilateral-trust REST API.
 */
export class TrustClient {
  constructor(private baseUrl: string = COGNILATERAL_API_BASE_URL) {
    this.baseUrl = baseUrl.replace(/\/+$/, '');
  }

  async summary(): Promise<TrustSummary> {
    const resp = await fetch(`${this.baseUrl}/api/trust/summary`);
    return resp.json();
  }

  async evaluate(confidence: number, opts?: {
    is_reversible?: boolean;
    touches_external?: boolean;
  }): Promise<TrustEvaluation> {
    const resp = await fetch(`${this.baseUrl}/api/trust/evaluate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ confidence, ...opts }),
    });
    return resp.json();
  }
}
