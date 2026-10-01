// ===== CHANGE THIS LINE TO SWITCH ENVIRONMENTS =====
const BASE_URL = 'https://ml.genproposals.com';  // PRODUCTION
// const BASE_URL = 'http://localhost:8000';           // LOCAL DEV - switch before deploy!
// ===================================================

// Main API endpoint
const API = `${BASE_URL}/api`;

// Specific API endpoints for each module
const AUTH_API = `${BASE_URL}/api/auth/verify`;
const JOB_DATA_API = `${BASE_URL}/api/job-data`;  // LEGACY - use JOBS_API for new code
const JOBS_API = `${BASE_URL}/api/jobs`;          // New job ingestion
const PROPOSALS_API = `${BASE_URL}/api/proposals`;
const ANALYTICS_API = `${BASE_URL}/api/analytics`;
const ADMIN_API = `${BASE_URL}/api/admin`;
const PROFILES_API = `${BASE_URL}/api/profiles`;  // Multi-tenant profiles
const PORTFOLIO_API = `${BASE_URL}/api/portfolio`;

// Tenant context (populated after auth)
let currentOrgId = null;
let currentUserId = null;
let currentProfileId = null;
