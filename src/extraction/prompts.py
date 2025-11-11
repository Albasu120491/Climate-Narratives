"""All prompt templates for AFA extraction."""

ACTOR_STANCE_PROMPT = """You are an expert analyst of financial news discourse.

TASK: Extract all actors and their stances on climate issues from the article below.

ARTICLE:
Headline: {headline}

{text}

INSTRUCTIONS:
1. Identify ALL actors mentioned (companies, financial institutions, governments, NGOs, individuals)
2. For each actor, determine their stance toward climate action:
   - supportive: Endorses climate action, investment, or regulation
   - opposing: Resists climate measures or highlights negative impacts
   - neutral: Mentions climate without evaluative judgment
   - mixed: Expresses both supportive and opposing views
3. Provide a DIRECT QUOTE from the article supporting each stance
4. Verify the statement is genuinely about climate (not just mentioning an energy company)

ACTOR TYPES:
- company: Corporate entities (e.g., ExxonMobil, Tesla)
- financial_institution: Banks, asset managers, investors (e.g., BlackRock, Goldman Sachs)
- government: Government bodies, regulators, officials (e.g., SEC, European Commission)
- ngo: NGOs, advocacy groups (e.g., Greenpeace, World Wildlife Fund)
- individual: Named individuals (researchers, experts, CEOs)

OUTPUT FORMAT (JSON):
{{
  "actors": [
    {{
      "name": "string",
      "actor_type": "company|financial_institution|government|ngo|individual",
      "stance": "supportive|opposing|neutral|mixed",
      "quote_text": "verbatim quote from article",
      "climate_relevance": "brief explanation of climate connection"
    }}
  ]
}}

IMPORTANT:
- Return ONLY valid JSON, no additional text
- If no actors found, return {{"actors": []}}
- Quotes must be verbatim from the article text
- Focus on explicit statements about climate, not just energy/environment generally

OUTPUT:"""

FRAME_CLASSIFICATION_PROMPT = """You are an expert in media framing analysis.

TASK: Classify how this article frames climate change.

ARTICLE:
Headline: {headline}

{text}

FRAME DEFINITIONS:
1. economic_opportunity: Climate action as growth, innovation, and investment potential
2. economic_risk: Financial losses, stranded assets, risks to firms or markets
3. regulatory_compliance: Laws, policies, and regulatory burdens or incentives
4. technological_solution: Innovation, R&D, and technical fixes to climate challenges
5. environmental_urgency: Ecological severity and need for rapid action
6. social_responsibility: Ethics, corporate responsibility, societal expectations
7. market_dynamics: Competition, supply-demand, market positioning
8. uncertainty_skepticism: Doubt about climate science, policies, or impacts

INSTRUCTIONS:
1. Assign ONE primary frame (the dominant way climate is presented)
2. Optionally assign ONE secondary frame (if clearly present alongside primary)
3. Justify your choice with specific evidence from the text
4. Verify this is genuinely about climate change (not just energy or economics generally)

EXAMPLES:
- "Renewable investments unlock new profit streams" → economic_opportunity
- "Carbon regulations will burden manufacturers" → regulatory_compliance (primary) + economic_risk (secondary)
- "Scientists warn of irreversible damage" → environmental_urgency

OUTPUT FORMAT (JSON):
{{
  "primary_frame": "frame_name",
  "secondary_frame": "frame_name_or_null",
  "justification": "explanation with brief quotes",
  "climate_connection": "how is this specifically about climate change?"
}}

IMPORTANT:
- Return ONLY valid JSON
- Primary frame is REQUIRED
- Secondary frame is OPTIONAL (use null if not present)
- Frame names must exactly match the options above

OUTPUT:"""

ARGUMENT_EXTRACTION_PROMPT = """You are an expert in argumentation analysis.

TASK: Extract the argumentative structure from this article.

ARTICLE:
Headline: {headline}

{text}

INSTRUCTIONS:
Extract the following components:

1. CLAIM: The main proposition or assertion being made about climate
2. EVIDENCE: Facts, statistics, expert testimony, or data supporting the claim
3. WARRANT: The logical reasoning connecting the evidence to the claim (may be implicit)
4. IMPACT: Stated or implied consequences if the claim is accepted
5. SUPPORTING ARGUMENTS: Any additional claim-evidence-warrant structures (optional)

EXAMPLE:
Text: "BlackRock argues that green portfolios outperform traditional funds. ESG funds gained 4.3% in 2020, compared to 2.1% for conventional indexes. This shows sustainable investing offers competitive advantages."

Output:
{{
  "claim": "Green portfolios outperform traditional investment funds",
  "evidence": ["ESG funds gained 4.3% in 2020", "Conventional indexes gained only 2.1%"],
  "warrant": "Higher returns demonstrate competitive advantages of sustainable investing",
  "impact": "Investors should prioritize ESG funds for better performance",
  "supporting_arguments": []
}}

OUTPUT FORMAT (JSON):
{{
  "claim": "main proposition",
  "evidence": ["evidence piece 1", "evidence piece 2"],
  "warrant": "logical connection between claim and evidence",
  "impact": "consequences or implications",
  "supporting_arguments": [
    {{
      "claim": "supporting claim",
      "evidence": ["evidence"],
      "warrant": "warrant"
    }}
  ]
}}

IMPORTANT:
- Return ONLY valid JSON
- If no clear argument, return {{"claim": null, "evidence": [], "warrant": null, "impact": null, "supporting_arguments": []}}
- Evidence should be specific quotes or data points
- Warrant explains WHY evidence supports the claim

OUTPUT:"""

# DVF Judge Prompts
DVF_COMPLETENESS_PROMPT = """Evaluate the COMPLETENESS of this extraction.

ORIGINAL ARTICLE:
{article_text}

EXTRACTION:
{extraction_json}

TASK: Score each dimension from 0.0 (complete failure) to 1.0 (perfect).

COMPLETENESS CHECKS:
1. actors: Are ALL actors mentioned in the article identified in the extraction?
2. stance: Is a stance provided for each actor?
3. frames: Are primary and secondary frames assigned appropriately?
4. arguments: Is the argument structure (claim, evidence, warrant) fully captured?

SCORING GUIDE:
- 1.0 = All components present and complete
- 0.7-0.9 = Most components present, minor omissions
- 0.4-0.6 = Some components present, significant gaps
- 0.1-0.3 = Few components present, major omissions
- 0.0 = No relevant components extracted

OUTPUT FORMAT (JSON):
{{
  "completeness": {{
    "actors": 0.0-1.0,
    "stance": 0.0-1.0,
    "frames": 0.0-1.0,
    "arguments": 0.0-1.0
  }},
  "explanation": "brief justification for scores"
}}

Return ONLY valid JSON.

OUTPUT:"""

DVF_FAITHFULNESS_PROMPT = """Evaluate the FAITHFULNESS of this extraction to the source text.

ORIGINAL ARTICLE:
{article_text}

EXTRACTION:
{extraction_json}

TASK: Verify that extracted content aligns with the article.

FAITHFULNESS CHECKS:
1. quote_alignment: Do all quotes exist verbatim in the article? (0.0-1.0)
2. paraphrase_equivalence: Are paraphrased claims semantically equivalent to source? (0.0-1.0)
3. no_hallucination: Is there any fabricated information not in the article? (0.0-1.0)

SCORING GUIDE:
- 1.0 = Perfect alignment, no hallucinations
- 0.7-0.9 = Minor paraphrasing discrepancies
- 0.4-0.6 = Some claims not well-supported
- 0.1-0.3 = Significant hallucinations or misrepresentations
- 0.0 = Completely unfaithful to source

OUTPUT FORMAT (JSON):
{{
  "faithfulness": {{
    "quote_alignment": 0.0-1.0,
    "paraphrase_equivalence": 0.0-1.0,
    "no_hallucination": 0.0-1.0
  }},
  "explanation": "specific issues found (if any)"
}}

Return ONLY valid JSON.

OUTPUT:"""

DVF_COHERENCE_PROMPT = """Evaluate the COHERENCE of this extraction.

EXTRACTION:
{extraction_json}

TASK: Check structural consistency and logical connections.

COHERENCE CHECKS:
1. schema_wellformed: Is the JSON properly formatted with all required fields? (0.0-1.0)
2. actor_frame_links: Do actor stances align logically with frames? (0.0-1.0)
3. frame_argument_links: Does the argument support the assigned frames? (0.0-1.0)
4. internal_consistency: Are there contradictions within the extraction? (0.0-1.0)

EXAMPLES:
- Actor with "supportive" stance + "economic_opportunity" frame = coherent
- Actor with "opposing" stance + "economic_opportunity" frame = potentially incoherent
- Claim about risks + "economic_opportunity" frame = incoherent

OUTPUT FORMAT (JSON):
{{
  "coherence": {{
    "schema_wellformed": 0.0-1.0,
    "actor_frame_links": 0.0-1.0,
    "frame_argument_links": 0.0-1.0,
    "internal_consistency": 0.0-1.0
  }},
  "explanation": "coherence issues identified"
}}

Return ONLY valid JSON.

OUTPUT:"""

DVF_RELEVANCE_PROMPT = """Evaluate the CLIMATE RELEVANCE of this extraction.

ARTICLE:
{article_text}

EXTRACTION:
{extraction_json}

TASK: Verify this is genuinely about climate change.

RELEVANCE CHECKS:
1. climate_focus: Is the extraction focused on climate change (not just energy/environment)? (0.0-1.0)
2. peripheral_excluded: Are non-climate elements appropriately excluded? (0.0-1.0)
3. frame_appropriateness: Are frames genuinely climate-related? (0.0-1.0)

EXAMPLES:
- "Solar panel manufacturer expands" → May not be climate-focused (just energy business)
- "Solar adoption to reduce emissions" → Climate-focused
- "Oil prices rise" → Not climate-relevant
- "Carbon tax on oil" → Climate-relevant

OUTPUT FORMAT (JSON):
{{
  "relevance": {{
    "climate_focus": 0.0-1.0,
    "peripheral_excluded": 0.0-1.0,
    "frame_appropriateness": 0.0-1.0
  }},
  "explanation": "relevance assessment"
}}

Return ONLY valid JSON.

OUTPUT:"""
