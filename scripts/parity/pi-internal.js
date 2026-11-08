// Functions that @earendil-works/pi-ai@1.5.0 does not export from its
// package entry point; re-exported from the installed artifact's dist/
// so vectors are generated from the published package on any machine.
// This is a .js shim so `tsc --noEmit` never descends into the dist
// sources.
export {
  splitErrorToolResultImages,
  transformMessages,
} from "../../node_modules/@earendil-works/pi-ai/dist/api/transform-messages.js";
export { sanitizeSurrogates } from "../../node_modules/@earendil-works/pi-ai/dist/utils/sanitize-unicode.js";
export { ownedSlice } from "../../node_modules/@earendil-works/pi-ai/dist/utils/text.js";
export {
  calculateContextTokens,
  estimateContextTokens,
  estimateMessageTokens,
  estimateTextAndImageContentTokens,
  estimateTextTokens,
} from "../../node_modules/@earendil-works/pi-ai/dist/utils/estimate.js";
export {
  adjustMaxTokensForThinking,
  clampMaxTokensToContext,
  clampReasoning,
  clampThinkingBudgetToAnswerRoom,
  resolveSamplingParams,
  thinkingBudgetForLevel,
  DEFAULT_THINKING_BUDGETS,
} from "../../node_modules/@earendil-works/pi-ai/dist/api/simple-options.js";
