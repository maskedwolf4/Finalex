/**
 * The one API base URL used by every browser request.  Set NEXT_PUBLIC_API_URL
 * for deployed environments; local development defaults to the FastAPI server.
 */
export const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");
