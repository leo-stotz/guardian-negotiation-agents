"""Buyer Agent Implementation"""

import json
import re
from typing import Dict, List, Any, Optional, Union
from agenticpay.agents.base_agent import BaseAgent
from agenticpay.agents.structured_message import extract_message_content
from agenticpay.models.base_llm import BaseLLM
from agenticpay.models.base_vlm import BaseVLM
from agenticpay.utils.user_profile import UserProfile, StylePreference, ShoppingHabit
from agenticpay.utils.thesis_utility import (
    agenticpay_contract_to_thesis,
    calculate_buyer_utility,
    validate_buyer_max_price,
    validate_buyer_reservation_utility,
    validate_value_ranges,
)
from loguru import logger


class ThesisBuyerAgent(BaseAgent):
    """Buyer Agent
    
    Represents the buyer, negotiates with the seller based on user requirements and budget.
    """
    
    def __init__(
        self,
        model: Union[BaseLLM, BaseVLM],
        name: str = "Buyer",
        role_description: str = "You are a buyer looking for a good deal.",
        buyer_max_price: Optional[float] = None,
        system_prompt_suffix: Optional[str] = None,
    ):
        """Initialize Buyer Agent
        
        Args:
            model: LLM or VLM interface (supports both BaseLLM and BaseVLM)
            name: Agent name
            role_description: Role description
            buyer_max_price: Maximum acceptable purchase price for buyer (bottom price, confidential information)
            system_prompt_suffix: Additional text to append to system prompt (e.g., personality profile)
        """
        super().__init__(model, role_description, name)
        self.buyer_max_price = buyer_max_price
        self.system_prompt_suffix = system_prompt_suffix
        self.last_selected_seller: Optional[int] = None
        # Populated by _evaluate_thesis_contract() from the most recently proposed
        # <contract> block, using thesis_utility.py exclusively (see there for schema).
        self.last_thesis_contract: Optional[Dict[str, float]] = None
        self.last_buyer_utility: Optional[float] = None
        self.last_validation: Optional[Dict[str, bool]] = None
        # Populated in respond() from the most recent <mental_model> block, for
        # components (e.g. the Guardian) that are explicitly allowed to read it.
        self.last_mental_model: Optional[str] = None
    
    def respond(
        self,
        conversation_history: List[Dict[str, Any]],
        current_state: Dict[str, Any],
    ) -> str:
        """Generate Buyer response
        
        Args:
            conversation_history: Conversation history
            current_state: Current state
            
        Returns:
            Buyer's response text
        """
        if not self.initialized:
            raise ValueError("Agent not initialized. Call initialize() first.")
        
        prompt = self._build_prompt(conversation_history, current_state)
        self.last_selected_seller = None
        
        # Get buyer's maximum acceptable price (bottom price)
        max_price = self.buyer_max_price or self.context.get('max_price', 'unknown')
        
        # Get product information (similar to seller_agent)
        product_info = self.context.get('product_info', {})
        available_products = self.context.get('available_products', [])
        
        # Format available products information
        available_products_info = ""
        if available_products:
            available_products_info = "\n\nAVAILABLE PRODUCTS IN YOUR INVENTORY:\n"
            for i, prod in enumerate(available_products, 1):
                available_products_info += f"{i}. {prod.get('name', 'Unknown')} - "
                available_products_info += f"Brand: {prod.get('brand', 'N/A')}, "
                available_products_info += f"Price: ${prod.get('price', 0):.2f}, "
                available_products_info += f"Features: {', '.join(prod.get('features', []))}\n"
            available_products_info += "\nYou can suggest other products from your inventory if they better match the buyer's needs.\n"
        
        # Get user profile information
        user_profile = self.context.get('user_profile')
        
        # Build user preference-related guidance
        preference_guidance = ""
        if user_profile:
            preference_guidance = "\nUSER PREFERENCES:\n"
            
            # If user_profile is a string (text description), use it directly
            if isinstance(user_profile, str):
                preference_guidance += f"- {user_profile}\n"
            # If user_profile is a UserProfile object, use the original logic
            elif isinstance(user_profile, UserProfile):
                # Style preference guidance
                if user_profile.style_preference:
                    style = user_profile.style_preference
                    if style == StylePreference.SIMPLE:
                        preference_guidance += "- Style preference: You prefer SIMPLE/MINIMALIST styles. Focus on clean, simple designs without excessive decoration.\n"
                    elif style == StylePreference.BUSINESS:
                        preference_guidance += "- Style preference: You prefer BUSINESS/PROFESSIONAL styles. Focus on formal, professional-looking items suitable for work.\n"
                    elif style == StylePreference.TRADITIONAL:
                        preference_guidance += "- Style preference: You prefer TRADITIONAL/CLASSIC styles. Focus on timeless, classic designs with traditional elements.\n"
                
                # Shopping habit guidance
                if user_profile.shopping_habit:
                    habit = user_profile.shopping_habit
                    if habit == ShoppingHabit.COMPARE:
                        preference_guidance += "- Shopping habit: You like to COMPARE PRICES and shop around. You may mention that you're comparing options, ask for better deals, or reference other sellers. Take your time in negotiations.\n"
                    elif habit == ShoppingHabit.DIRECT:
                        preference_guidance += "- Shopping habit: You prefer DIRECT PURCHASES. You value efficiency and may be willing to pay a fair price quickly if the deal is reasonable. Don't waste too much time haggling.\n"
        
        # Add Buyer-specific guidance
#         buyer_guidance = f"""

# IMPORTANT REMINDERS:
# - Your maximum acceptable price (confidential - do not reveal this to the seller) is ${max_price}
# - You want to negotiate a fair price that fits your needs
# - Consider the environment factors: {self.context.get('environment_info', {})}
# - Be polite but firm in your negotiations
# - Try to find a win-win solution
# - If the seller's price is too high, suggest a reasonable counter-offer
# - Try to negotiate the price as low as possible, but ensure the deal is successful in the end
# - **CRITICAL: Each conversation you MUST make one price offer, you MUST use the format: ### BUYER_PRICE($X) ###**
# - Example: "I can offer ### BUYER_PRICE($100) ### for this product"
# - Example: "How about ### BUYER_PRICE($120.50) ###?"
# - This specific format is required for the system to correctly extract your offer price
# - NEVER reveal your maximum acceptable price to the seller - keep it confidential
# - Keep communication short and concise.

# DEAL AGREEMENT INSTRUCTION:
# - If you decide to accept the deal and want to make a transaction, you MUST include the exact phrase "MAKE_DEAL" in your response
# - This phrase should appear when you are ready to finalize the agreement
# - Example: "That sounds good! I accept your offer. MAKE_DEAL"
# - Only use "MAKE_DEAL" when you are genuinely ready to complete the transaction
# {preference_guidance}
# Now, respond as {self.name}:
# """

        # Add personality profile if provided
        personality_section = ""
        if self.system_prompt_suffix:
            personality_section = f"\n{self.system_prompt_suffix}\n"

        num_sellers = current_state.get("num_sellers")
        if num_sellers is None:
            num_sellers = self.context.get("num_sellers", 1)
        try:
            num_sellers = int(num_sellers)
        except (TypeError, ValueError):
            num_sellers = 1
        num_sellers = max(1, num_sellers)

        task_instruction = current_state.get("instruction", "")
        task_instruction_section = ""
        if task_instruction:
            task_instruction_section = f"\nTASK INSTRUCTION:\n- {task_instruction}\n"
        guardian_feedback = current_state.get("guardian_feedback", "")
        guardian_feedback_section = ""
        if guardian_feedback:
            guardian_feedback_section = f"""
INTERNAL REVIEW FEEDBACK (confidential, not visible to the seller):
Your previous draft for this round was reviewed internally before being sent and needs revision:
{guardian_feedback}
Revise your proposal to address this feedback. You remain free to choose how — there is no
prescribed contract, price, or concession size. Provide a complete, revised proposal following
the same output format as before.
"""
        contract_config = (
            self.context.get("contract_config")
            or self.context.get("environment_info", {}).get("contract_config")
            or current_state.get("contract_config")
        )
        seller_contract_configs = (
            self.context.get("seller_contract_configs")
            or self.context.get("environment_info", {}).get("seller_contract_configs")
            or current_state.get("contract_configs")
        )
        message_reasoning_rule = (
            "MESSAGE RULE: Before giving your <contract>, briefly explain your reasoning. "
            "When relevant, mention what you observe in the product text or image."
        )

        selected_seller_rules = f"""
- MULTI-SELLER ROUTING: You are negotiating in a multi-seller setting.
- You MUST choose exactly one seller for this turn and output it in a dedicated `<selected_seller>` block.
- `<selected_seller>` must contain exactly one integer from `1` to `{num_sellers}`.
- If there is only one seller in the task, you MUST output `1`.
- Base your choice on the conversation history and current offers.
- Do NOT put any extra text inside `<selected_seller>`.
"""
        selected_seller_format = f"""<selected_seller>
[one integer from 1 to {num_sellers}]
</selected_seller>
"""
        action_format = ""

#         buyer_guidance = f"""
# IMPORTANT:
# - Your top price is ${max_price} (confidential, do not reveal).
# - Current product information: {product_info}
# {available_products_info}
# - Consider the environment: {self.context.get('environment_info', {})}.
# {personality_section}
# - **CRITICAL: In each turn, you MUST make exactly ONE price offer for the product using the format:**
#   ### BUYER_PRICE($X) ###
# - **IMPORTANT: BUYER_PRICE($X) must be the TOTAL PRICE for the entire order/transaction, NOT a per-unit price.**
#   - If ordering multiple units/items, $X should be the total amount you will pay.
#   - Example: For 10,000 units at $0.40 each, use ### BUYER_PRICE($4000) ###, NOT ### BUYER_PRICE($0.40) ###
# - Example: "I can offer ### BUYER_PRICE($10) ### for this product."
# - Example: "How about ### BUYER_PRICE($12.50) ###?"
# - This specific format is required for the system to correctly extract your offer price.
# - NEVER reveal your maximum acceptable price to the seller.
# - Keep communication short (150 words or less), clear, and focused on negotiation.

# DEAL AGREEMENT INSTRUCTION:
# - Only finalize the transaction when you believe the price is reasonably balanced.
# - If you decide to accept the deal, you MUST include the exact phrase "MAKE_DEAL" in your response.
# - Example: "That sounds acceptable to me. MAKE_DEAL"

# {preference_guidance}

# Now, respond as {self.name}:
# """

        if seller_contract_configs:
            config_summaries = {}
            for seller_id, cfg in seller_contract_configs.items():
                buyer_preferences = cfg.get("buyer_preferences", {})
                config_summaries[seller_id] = {
                    "field_descriptions": cfg.get("field_descriptions", {}),
                    "continuous_bounds": cfg.get("continuous_bounds", {}),
                    "discrete_options": cfg.get("discrete_options", {}),
                    "buyer_preferences": buyer_preferences,
                    "weight_descriptions": buyer_preferences.get("weight_descriptions", {}),
                    "contrainfo": cfg.get("contrainfo", {}),
                }
            contract_rules = f"""
CONTRACT INFORMATION:
- You are negotiating with multiple sellers. Each seller may have a different contract_config, but all sellers use the same JSON schema:
  {{
    "price": <number>,
    "continuous_terms": <object>,
    "discrete_terms": <object>
  }}
- Seller-specific public fields and your private buyer utility information: {config_summaries}
- No contract field has a default value or is pre-agreed at the start of negotiation.
- The `<contract>` JSON in your message is your current complete proposal for the selected seller, with every field filled by you.
- Each `<message>` must contain exactly one `<contract>...</contract>` JSON block following the selected seller's schema/options.
- Never reveal your confidential reservation value directly.
"""
            deal_instruction = """
DEAL AGREEMENT INSTRUCTION:
- A deal is finalized only when you and the selected seller output compatible contract JSONs.
- If you accept, repeat the accepted final contract JSON exactly in `<contract>...</contract>`.
"""
            message_instruction = f"[Your actual negotiation message to the selected seller. {message_reasoning_rule} You can discuss any contract terms, but must include exactly one valid <contract>...</contract> JSON block for that seller.]"
        elif contract_config:
            continuous_bounds = contract_config.get("continuous_bounds", {})
            discrete_options = contract_config.get("discrete_options", {})
            field_descriptions = contract_config.get("field_descriptions", {})
            buyer_preferences = contract_config.get("buyer_preferences", {})
            weight_descriptions = buyer_preferences.get("weight_descriptions", {})
            contra_info = contract_config.get("contrainfo") or self.context.get("environment_info", {}).get("contrainfo", {})
            contract_rules = f"""
CONTRACT INFORMATION:
- The negotiable contract has this JSON schema:
  {{
    "price": <number>,
    "continuous_terms": <object>,
    "discrete_terms": <object>
  }}
- Field meanings: {field_descriptions}
- Valid continuous terms and bounds: {continuous_bounds}
- Valid discrete terms and options: {discrete_options}
- Your private utility information: {buyer_preferences}
- Meaning of your private utility information: {weight_descriptions}
- Additional contract background (`contrainfo`): {contra_info}
- No contract field has a default value or is pre-agreed at the start of negotiation.
- The `<contract>` JSON in your message is your current complete proposal, with every field filled by you.
- Each `<message>` must contain exactly one `<contract>...</contract>` JSON block following the schema above.
- Never reveal your confidential reservation value directly.
"""
            deal_instruction = """
DEAL AGREEMENT INSTRUCTION:
- A deal is finalized only when both sides output compatible contract JSONs.
- Before your <contract>, state your intended action in a dedicated <action> block: exactly ACCEPT if you are accepting the seller's most recently offered contract exactly as-is, or COUNTER for any new or revised proposal (including one that only slightly differs from the seller's last offer -- COUNTER covers anything short of an unconditional exact acceptance).
- If ACCEPT, still repeat the accepted final contract JSON exactly in `<contract>...</contract>`. Note that when <action>ACCEPT</action> is detected, the system also enforces this automatically using the seller's last offer, regardless of what you write in <contract> -- so an ACCEPT always resolves to the seller's exact terms.
"""
            message_instruction = f"[Your actual negotiation message to the seller. {message_reasoning_rule} You can discuss any contract terms, but must include exactly one valid <contract>...</contract> JSON block.]"
            action_format = """<action>
[ACCEPT or COUNTER]
</action>
"""
        else:
            contract_rules = """
- **CRITICAL: In each turn, you MUST include exactly ONE ### BUYER_PRICE($X) ### inside `<message>` — including when you accept or confirm a deal.** There are no exceptions: if you only say you accept without this tag, the environment keeps your previous numeric offer and the deal will fail to register as agreed.
- When you accept the seller's price, set $X to the total you agree to pay (typically the seller's last stated price in ### SELLER_PRICE($Y) ### that you are accepting). When counter-offering, $X is your new offer.
- **IMPORTANT: BUYER_PRICE($X) must be the TOTAL PRICE for the entire order/transaction, NOT a per-unit price.**
  - If ordering multiple units/items, $X should be the total amount you will pay.
  - Example: For 10,000 units at $0.40 each, use ### BUYER_PRICE($4000) ###, NOT ### BUYER_PRICE($0.40) ###
- Example: "I can offer ### BUYER_PRICE($10) ### for this product."
- Example: "How about ### BUYER_PRICE($12.50) ###?"
- Example (accepting their price): "Deal — I'll take it at ### BUYER_PRICE($6.50) ###."
- This specific format is required for the system to correctly extract your offer price.
"""
            deal_instruction = """
DEAL AGREEMENT INSTRUCTION:
- Only finalize the transaction when you believe the price is reasonably balanced.
- When you accept, state it clearly in `<message>` and include ### BUYER_PRICE($X) ### with $X equal to the agreed total you will pay (same as the price you are accepting). Do not use any special deal-finalization keyword beyond that.
- Wrong: accepting verbally with no ### BUYER_PRICE(...) ### — the environment keeps your previous numeric offer and agreement may not register.
- Right: "I accept your offer at ### BUYER_PRICE($6.50) ###."
"""
            message_instruction = "[Your actual negotiation message to the seller. Must include exactly one ### BUYER_PRICE($X) ### and obey all IMPORTANT / DEAL AGREEMENT rules above.]"
        thesis_guidance = f"""
THESIS NEGOTIATION CONSTRAINTS:

HARD CONSTRAINTS:
- Your maximum willingness to pay (`max_price`) is a strict upper limit, not a target price. Never propose or accept a price above ${max_price}.
- Price, delivery time, and payment term must always remain within the permitted ranges defined in the scenario.
- Only accept a deal if your own total utility (`U_buyer`) is at least your reservation utility (`U_min`).
- Always comply with the required contract schema and output format defined below.
- Your `max_price`, utility weights, reservation utility, and any other explicitly confidential preference information are private.
- Never reveal or describe confidential preference information in the visible `<message>` sent to the seller.
- You may use confidential values internally in your reasoning and `<mental_model>`.

STRATEGIC GUIDELINES:
- You are responsible for choosing your negotiation strategy autonomously. There is no predefined formula for your opening offer or concession path.
- Choose your opening offer based on the current scenario, your preferences, and your negotiation objective.
- Aim to maximize your own total economic utility, not merely to reach an agreement.
- Evaluate offers as complete packages across price, delivery time, and payment term rather than focusing only on price.
- Use trade-offs between negotiation issues when they can improve your overall utility.
- Consider the seller's previous offers, concessions, and observable negotiation behavior when deciding your next move.
- Make concessions only when they are strategically justified. Avoid automatic, unnecessary, or excessively large concessions.
- Take the remaining negotiation rounds into account when deciding whether to continue negotiating, concede, or accept.
- A no-deal outcome is acceptable if no sufficiently beneficial agreement can be reached within your constraints.
- Do not repeat the same contract automatically when the seller changes its offer. Re-evaluate the full package each round.
- When the seller makes a concession on one issue, actively consider whether you can exchange movement on a less important issue for improvement on a more important one.
- Use changes in the seller's offers as information about where trade-offs may be possible.
- Explicitly revise your estimate of the opponent's reservation price whenever the seller's offer changes. Use observed concessions as evidence to update both the estimate and your confidence.
- Do not keep repeating an outdated reservation-price estimate across multiple rounds without justification.
"""
        buyer_guidance = f"""
IMPORTANT:
- Current product information: {product_info}
{available_products_info}
- Consider the environment: {self.context.get('environment_info', {})}.
{personality_section}
{task_instruction_section}
{thesis_guidance}
{contract_rules}
{guardian_feedback_section}
- NEVER reveal your maximum acceptable price to the seller.
{selected_seller_rules}

{deal_instruction}

{preference_guidance}

MENTAL MODELING INSTRUCTION:
Before replying, briefly state your internal assessment for evaluation:
1. [Opponent Reservation Price]: Output exactly `price=<number>; confidence=<0-100>` for your estimate of the opponent's reservation price and confidence score.
2. [Opponent Strategy]: Your inference about the opponent's tactic.
3. [My Strategy]: Your chosen tactic and reasoning.

You MUST format your entire output exactly as follows:
<mental_model>
[Opponent Reservation Price]: price=<number>; confidence=<0-100>
[Opponent Strategy]: <your inference about the opponent's tactic>
[My Strategy]: <your chosen tactic and reasoning>
</mental_model>
{selected_seller_format}{action_format}<message>
{message_instruction}
</message>
"""

        full_prompt = prompt + buyer_guidance

        # logger.info(f"Buyer prompt: {full_prompt}")
        
        # Extract images from current_state if VLM is used
        images = None
        if self.is_vlm:
            # Check for images in current_state (e.g., product images)
            images = current_state.get('images') or current_state.get('product_images')
            # Also check in context
            if images is None:
                images = self.context.get('images') or self.context.get('product_images')
        
        # Generate response: VLM supports images, LLM doesn't
        if self.is_vlm and images is not None:
            response = self.model.generate(
                full_prompt, 
                images=images,
                temperature=0.0,
                max_tokens=2048  # Ensure complete response generation
            )
        else:
            response = self.model.generate(
                full_prompt, 
                temperature=0.0,
                max_tokens=2048  # Increased to accommodate mental model + message
            )
        
        # Remove <think>...</think> tags (used by reasoning models like DeepSeek)
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)

        # Extract <mental_model> block — log it but do NOT include in returned message
        mental_model_match = re.search(r'<mental_model>(.*?)</mental_model>', response, flags=re.DOTALL | re.IGNORECASE)
        self.last_mental_model = mental_model_match.group(1).strip() if mental_model_match else None
        if mental_model_match:
            mental_model_content = mental_model_match.group(1).strip()
            round_num = current_state.get("current_round")
            if round_num is None:
                round_num = current_state.get("round")
            round_str = str(round_num) if round_num is not None else "?"
            logger.info(
                f"\n{'='*50}\n[{self.name} MENTAL MODEL | round {round_str}]\n{mental_model_content}\n{'='*50}"
            )

        selected_seller_match = re.search(r'<selected_seller>\s*(\d+)\s*</selected_seller>', response, flags=re.DOTALL | re.IGNORECASE)
        if selected_seller_match:
            parsed_selected_seller = int(selected_seller_match.group(1))
            if 1 <= parsed_selected_seller <= num_sellers:
                self.last_selected_seller = parsed_selected_seller

        # Extract <message> block — this is the only part that enters conversation history
        final_message = extract_message_content(response)
        if final_message is None:
            # No <message> tag: strip structured blocks and use remainder
            logger.warning(f"[{self.name}] Output did not follow the expected structured format. Using fallback.")
            final_message = re.sub(r'<mental_model>.*?</mental_model>', '', response, flags=re.DOTALL | re.IGNORECASE)
            final_message = re.sub(r'<selected_seller>.*?</selected_seller>', '', final_message, flags=re.DOTALL | re.IGNORECASE).strip()

        # Deterministic ACCEPT override (single-seller contract-mode only, i.e. the
        # `elif contract_config:` branch above, which is the only branch emitting an
        # <action> tag). AgenticPay's own _contracts_compatible() requires the two
        # sides' continuous_terms/discrete_terms to match EXACTLY (see
        # Task1_basic_price_negotiation._contracts_compatible) -- an LLM regenerating
        # the seller's numbers as free text cannot guarantee that. If the buyer
        # explicitly signals <action>ACCEPT</action>, the seller's most recently
        # offered contract is substituted verbatim for whatever <contract> the model
        # produced, so an intended acceptance always matches exactly. This runs
        # BEFORE _evaluate_thesis_contract() below (patching `response` too, not just
        # final_message), so every downstream consumer -- the compliance/Guardian
        # hook here, an external Guardian review, and the message actually sent to
        # the seller/env -- all see the identical, already-corrected contract.
        if contract_config and not seller_contract_configs:
            action_match = re.search(r'<action>\s*(ACCEPT|COUNTER)\s*</action>', response, flags=re.IGNORECASE)
            if action_match and action_match.group(1).upper() == "ACCEPT":
                seller_contract = self._extract_last_seller_contract(conversation_history)
                if seller_contract is not None:
                    corrected_block = f"<contract>{json.dumps(seller_contract, sort_keys=True)}</contract>"
                    contract_pattern = r'<contract>.*?</contract>'
                    if re.search(contract_pattern, final_message, flags=re.DOTALL | re.IGNORECASE):
                        final_message = re.sub(contract_pattern, corrected_block, final_message, count=1, flags=re.DOTALL | re.IGNORECASE)
                    else:
                        final_message = f"{final_message}\n{corrected_block}"
                    if re.search(contract_pattern, response, flags=re.DOTALL | re.IGNORECASE):
                        response = re.sub(contract_pattern, corrected_block, response, count=1, flags=re.DOTALL | re.IGNORECASE)
                    else:
                        response = f"{response}\n{corrected_block}"
                    logger.info(f"[{self.name}] <action>ACCEPT</action> detected -> substituted seller's exact contract: {seller_contract}")
                else:
                    logger.warning(f"[{self.name}] <action>ACCEPT</action> detected but no seller contract found in history; keeping model-generated contract.")

        # Evaluate the proposed <contract> (if any) exclusively via thesis_utility.py.
        # Runs on the (possibly ACCEPT-corrected) response, so it always reflects
        # what is actually returned below.
        self._evaluate_thesis_contract(response, current_state)

        return final_message

    def _extract_last_seller_contract(self, conversation_history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Extract the most recent seller `<contract>` block (AgenticPay wire format).

        Uses the same `<contract>...</contract>` extraction convention as
        elsewhere in the thesis code, duplicated here rather than imported --
        AgenticPay's own `_extract_contract` is a private env method, not a
        public utility (see thesis_rule_based_seller.py's module docstring for
        the same rationale). Only used for the deterministic ACCEPT override.
        """
        for message in reversed(conversation_history):
            if message.get("role") != "seller":
                continue
            match = re.search(r"<contract>\s*(.*?)\s*</contract>", message.get("content", "") or "", re.DOTALL | re.IGNORECASE)
            if not match:
                continue
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                return None
        return None

    def _evaluate_thesis_contract(self, response: str, current_state: Dict[str, Any]) -> None:
        """Extract the proposed `<contract>` JSON from the raw model response (if
        present) and evaluate it using thesis_utility.py exclusively: adapts it to
        the thesis-internal schema, computes the buyer utility, and validates value
        ranges / max_price / reservation utility.

        Populates `self.last_thesis_contract` / `self.last_buyer_utility` /
        `self.last_validation` for downstream use (e.g. Guardian/Evaluator, tests)
        and logs a summary. Does not alter the negotiation message or raise on a
        malformed/absent contract.
        """
        self.last_thesis_contract = None
        self.last_buyer_utility = None
        self.last_validation = None

        contract_match = re.search(r"<contract>\s*(.*?)\s*</contract>", response, re.DOTALL | re.IGNORECASE)
        if not contract_match:
            return

        try:
            agenticpay_contract = json.loads(contract_match.group(1).strip())
            thesis_contract = agenticpay_contract_to_thesis(agenticpay_contract)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(f"[{self.name}] Could not evaluate proposed contract via thesis_utility: {exc}")
            return

        buyer_utility = calculate_buyer_utility(thesis_contract)
        range_ok = validate_value_ranges(thesis_contract)
        reservation_ok = validate_buyer_reservation_utility(buyer_utility)

        max_price_ok = True
        try:
            buyer_max_price = float(self.buyer_max_price)
            max_price_ok = validate_buyer_max_price(thesis_contract["price"], buyer_max_price)
        except (TypeError, ValueError):
            pass  # no numeric buyer_max_price configured -> nothing to check

        self.last_thesis_contract = thesis_contract
        self.last_buyer_utility = buyer_utility
        self.last_validation = {
            "range_ok": range_ok,
            "max_price_ok": max_price_ok,
            "reservation_ok": reservation_ok,
        }

        round_num = current_state.get("current_round", current_state.get("round"))
        logger.info(
            f"[{self.name} THESIS UTILITY | round {round_num}] contract={thesis_contract} "
            f"buyer_utility={buyer_utility:.4f} range_ok={range_ok} "
            f"max_price_ok={max_price_ok} reservation_ok={reservation_ok}"
        )
        if not (range_ok and max_price_ok and reservation_ok):
            logger.warning(
                f"[{self.name}] Proposed contract fails thesis validation: "
                f"range_ok={range_ok} max_price_ok={max_price_ok} reservation_ok={reservation_ok}"
            )

