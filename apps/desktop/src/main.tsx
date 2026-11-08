import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { App } from "./app.tsx";

import "./styles.css";

const root = document.querySelector("#app");
if (!root) {
  throw new Error("Missing #app");
}

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
