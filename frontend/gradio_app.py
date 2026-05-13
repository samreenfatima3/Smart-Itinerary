"""
Smart Travel AI - Professional Itinerary Generator
Enhanced with maps, images, and interactive elements
"""

from __future__ import annotations

import os
import json
import base64
import random
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from urllib.parse import urlencode

import gradio as gr
import requests
import plotly.graph_objects as go
import plotly.express as px

# Configuration
DEFAULT_API_URL = "http://127.0.0.1:8000"
API_BASE = os.getenv("ITINERARY_API_URL", DEFAULT_API_URL).rstrip("/")
ITINERARY_ENDPOINT = "/itinerary/generate"
CHATBOT_ENDPOINT = "/chatbot/chat"
USER_REGISTER_ENDPOINT = "/user/register"
USER_LOGIN_ENDPOINT = "/user/login"
USER_PROFILE_ENDPOINT = "/user/profile"

# Professional theme with travel-inspired colors
THEME = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="blue",
    neutral_hue="slate"
).set(
    button_primary_background_fill="linear-gradient(135deg, #0d9488, #0891b2)",
    button_primary_background_fill_hover="linear-gradient(135deg, #0f766e, #0e7490)",
    button_primary_text_color="white",
    block_background_fill="*background_fill_primary",
    block_border_width="0",
    block_shadow="0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)"
)

CUSTOM_CSS = """
:root {
    --hero-gradient: linear-gradient(135deg, #0d9488 0%, #0891b2 50%, #0369a1 100%);
    --card-shadow: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 10px 10px -5px rgb(0 0 0 / 0.04);
    --border-glow: 0 0 20px rgba(13, 148, 136, 0.15);
}

.gradio-container {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%) !important;
}

/* Fix text color issues - ensure all text is visible */
* {
    color: inherit;
}

label, .label-wrap, .form, .form > *, 
.wrap label, .wrap .label-wrap,
.gr-form label, .gr-textbox label, 
.gr-dropdown label, .gr-radio label, 
.gr-slider label, .gr-accordion label {
    color: #1e293b !important;
}

/* Fix all text inputs and textareas */
input[type="text"], input[type="number"], 
textarea, select {
    color: #1e293b !important;
    background-color: #ffffff !important;
}

.gr-textbox input, .gr-textbox textarea,
.gr-dropdown select, .gr-dropdown input {
    color: #1e293b !important;
    background-color: #ffffff !important;
}

/* Fix markdown text */
.gr-markdown, .gr-markdown *, 
.gr-markdown p, .gr-markdown h1, 
.gr-markdown h2, .gr-markdown h3,
.gr-markdown h4, .gr-markdown h5,
.gr-markdown h6, .gr-markdown li,
.gr-markdown ul, .gr-markdown ol {
    color: #1e293b !important;
}

/* Fix all paragraph and span text */
p, span, div, h1, h2, h3, h4, h5, h6 {
    color: inherit;
}

/* Override white text in specific contexts */
.block, .block * {
    color: inherit;
}

.gr-group {
    padding: 1.5rem !important;
    margin-bottom: 1.5rem !important;
    background: white !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1) !important;
    gap: 1rem !important;
}

.gr-textbox, .gr-dropdown, .gr-radio, .gr-slider, .gr-image {
    margin-bottom: 1.25rem !important;
    margin-top: 0.5rem !important;
}

/* Add spacing between form elements */
.gr-form > * {
    margin-bottom: 1rem !important;
}

/* Ensure proper spacing in groups */
.gr-group > *:not(:last-child) {
    margin-bottom: 1rem !important;
}

/* Fix info text visibility */
.gr-textbox .info, .gr-dropdown .info, .gr-radio .info, .gr-slider .info {
    color: #64748b !important;
}

/* Fix status textbox */
.gr-textbox[type="text"] {
    color: #1e293b !important;
}

/* Fix all input placeholders */
input::placeholder, textarea::placeholder {
    color: #94a3b8 !important;
    opacity: 1 !important;
}

/* Ensure all text in forms is visible */
.gr-form, .gr-form * {
    color: #1e293b !important;
}

/* Fix accordion text */
.gr-accordion, .gr-accordion label {
    color: #1e293b !important;
}

/* Fix image labels */
.gr-image label {
    color: #1e293b !important;
}

/* Fix status display */
.gr-textbox input[readonly] {
    color: #1e293b !important;
    background-color: #f1f5f9 !important;
}

/* Ensure images display properly */
.gr-image, .gr-image * {
    color: #1e293b !important;
}

.gr-image img {
    border-radius: 8px !important;
    object-fit: cover !important;
    width: 100% !important;
    height: auto !important;
    display: block !important;
    max-width: 100% !important;
}

.gr-image .wrap, .gr-image .wrap-inner {
    display: block !important;
    width: 100% !important;
}

/* Fix image container */
.gr-image > div, .gr-image > div > div {
    width: 100% !important;
    display: block !important;
}

/* Ensure image loads even if URL fails */
.gr-image img[src=""], .gr-image img:not([src]) {
    background: linear-gradient(135deg, #e2e8f0, #cbd5e1) !important;
    min-height: 150px !important;
}

/* Fix any remaining white text issues */
.gr-tabs, .gr-tabs *, .gr-tabs label {
    color: #1e293b !important;
}

/* Fix component wrappers */
.wrap, .wrap-inner, .wrap-inner * {
    color: #1e293b !important;
}

/* Fix status and output textboxes */
.gr-textbox input[readonly], 
.gr-textbox textarea[readonly] {
    color: #1e293b !important;
    background-color: #f8fafc !important;
}

/* Fix radio button labels */
.gr-radio label, 
.gr-radio .wrap label,
input[type="radio"] + label {
    color: #1e293b !important;
}

/* Fix dropdown options */
select option {
    color: #1e293b !important;
    background-color: #ffffff !important;
}

/* Ensure all visible text is dark */
.visible, .visible * {
    color: #1e293b !important;
}

/* Fix chatbot text - make it white for better visibility */
.gr-chatbot, .gr-chatbot *, 
.gr-chatbot .message, .gr-chatbot .message *,
.gr-chatbot .user, .gr-chatbot .user *,
.gr-chatbot .assistant, .gr-chatbot .assistant *,
.gr-chatbot .wrap, .gr-chatbot .wrap *,
.gr-chatbot div, .gr-chatbot span, .gr-chatbot p,
.gr-chatbot .md, .gr-chatbot .chatbot, .gr-chatbot .prose,
.gr-chatbot .message-content, .gr-chatbot .message-content *,
.gr-chatbot .bubble, .gr-chatbot .bubble *,
.gr-chatbot [class*="svelte"], .gr-chatbot [class*="svelte"] * {
    color: white !important;
}

/* Ensure chatbot container text is white */
.gr-chatbot label, .gr-chatbot .label-wrap {
    color: white !important;
}

/* Fix specific chatbot message styling */
.gr-chatbot .message-content span,
.gr-chatbot .message-content p,
.gr-chatbot .message-content ul,
.gr-chatbot .message-content li,
.gr-chatbot .message-content strong,
.gr-chatbot .message-content h1,
.gr-chatbot .message-content h2,
.gr-chatbot .message-content h3 {
    color: white !important;
}

/* Fix chatbot bubbles background if needed */
.gr-chatbot .user-row, .gr-chatbot .bot-row {
    color: white !important;
}

/* Fix JSON display */
.gr-json, .gr-json * {
    color: #1e293b !important;
    background-color: #ffffff !important;
}

/* Global override to ensure all text is visible - catch any remaining white text */
/* Target specific Gradio components that might have white text */
.block:not(#hero-section):not(.stats-card) *,
.wrap:not(#hero-section):not(.stats-card) *,
.form:not(#hero-section):not(.stats-card) * {
    color: #1e293b !important;
}

/* Exception for hero section which should have white text */
#hero-section, #hero-section *,
#hero-section .hero-title, #hero-section .hero-subtitle,
#hero-section .stats-card, #hero-section .stats-number,
#hero-section .stats-label {
    color: white !important;
}

.stats-card, .stats-card *,
.stats-card .stats-number, .stats-card .stats-label {
    color: white !important;
}

/* Ensure inputs and selects have visible text */
input:not([type="submit"]):not([type="button"]):not([type="radio"]):not([type="checkbox"]),
textarea, select {
    color: #1e293b !important;
}

/* Fix button text - keep primary buttons white, others dark */
button:not(.cta-button):not([variant="primary"]),
.gr-button:not(.cta-button):not([variant="primary"]) {
    color: #1e293b !important;
}

/* Beautiful animations for visualizations */
.gr-plot, .gr-plot * {
    animation: fadeInUp 0.6s ease-out;
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Smooth transitions for plot updates */
.plotly {
    transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

/* Enhanced visualization container */
.gr-tabs .gr-plot {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
    background: white;
    padding: 1rem;
    margin: 1rem 0;
}

/* Hover effects for charts */
.gr-plot:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
    transition: all 0.3s ease;
}

#hero-section {
    background: var(--hero-gradient);
    color: white;
    padding: 3rem 2rem;
    border-radius: 24px;
    margin: 1rem 0;
    box-shadow: var(--card-shadow);
    border: 1px solid rgba(255, 255, 255, 0.2);
    position: relative;
    overflow: hidden;
}

#hero-section::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: url("data:image/svg+xml,%3Csvg width='100' height='100' viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M11 18c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm48 25c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm-43-7c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm63 31c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM34 90c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm56-76c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM12 86c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm28-65c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm23-11c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-6 60c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm29 22c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zM32 63c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm57-13c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-9-21c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM60 91c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM35 41c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM12 60c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2z' fill='%23ffffff' fill-opacity='0.1' fill-rule='evenodd'/%3E%3C/svg%3E");
    opacity: 0.3;
}

.hero-content {
    position: relative;
    z-index: 2;
    text-align: center;
}

.hero-title {
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    margin-bottom: 1rem !important;
    background: linear-gradient(135deg, #ffffff 0%, #e2e8f0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-subtitle {
    font-size: 1.2rem !important;
    opacity: 0.9;
    margin-bottom: 2rem !important;
}

.feature-card {
    background: white;
    border-radius: 16px;
    padding: 1.5rem;
    margin: 0.5rem 0;
    box-shadow: var(--card-shadow);
    border: 1px solid #e2e8f0;
    transition: all 0.3s ease;
}

.feature-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 25px 50px -12px rgb(0 0 0 / 0.25);
}

.stats-card {
    background: linear-gradient(135deg, #0f766e, #0891b2);
    color: white;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    margin: 0.5rem;
}

.stats-number {
    font-size: 1.8rem;
    font-weight: 700;
    display: block;
}

.stats-label {
    font-size: 0.9rem;
    opacity: 0.9;
}

.itinerary-card {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: var(--card-shadow);
    border-left: 4px solid #0d9488;
}

.day-highlight {
    background: linear-gradient(135deg, #f0fdfa, #ecfeff);
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    border: 1px solid #a5f3fc;
}

.budget-breakdown {
    background: #f8fafc;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}

.progress-bar {
    height: 8px;
    background: #e2e8f0;
    border-radius: 4px;
    margin: 0.5rem 0;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #0d9488, #0891b2);
    border-radius: 4px;
    transition: width 0.5s ease;
}

.map-container {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: var(--card-shadow);
}

.chat-message {
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 12px;
    max-width: 80%;
}

.user-message {
    background: linear-gradient(135deg, #0d9488, #0891b2);
    color: white;
    margin-left: auto;
}

.assistant-message {
    background: #f1f5f9;
    color: #334155;
}

.cta-button {
    background: linear-gradient(135deg, #0d9488, #0891b2) !important;
    border: none !important;
    color: white !important;
    padding: 1rem 2rem !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    transition: all 0.3s ease !important;
}

.cta-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 10px 10px -5px rgb(0 0 0 / 0.04);
}
"""

# Sample destination images and coordinates (in production, you'd fetch these from an API)
# Using None for images - will be set dynamically when destination is entered
# This avoids network dependency issues
DESTINATION_IMAGES = {
    "skardu": None,
    "hunza": None,
    "naran": None,
    "swat": None,
    "murree": None,
    "gilgit": None,
    "default": None
}

# Popular destination coordinates in Pakistan
DESTINATION_COORDS = {
    "skardu": (35.2971, 75.6333),
    "hunza": (36.3167, 74.6500),
    "naran": (34.9100, 73.6500),
    "swat": (35.2200, 72.4200),
    "murree": (33.9072, 73.3903),
    "gilgit": (35.9208, 74.3083),
    "islamabad": (33.6844, 73.0479),
    "karachi": (24.8607, 67.0011),
    "lahore": (31.5497, 74.3436),
    "peshawar": (34.0151, 71.5249),
    "quetta": (30.1798, 66.9750),
    "multan": (30.1575, 71.5249),
    "faisalabad": (31.4504, 73.1350),
}

# Common places/attractions coordinates in Pakistan
# Using more accurate coordinates for better map plotting
PLACE_COORDS = {
    # Skardu area - more specific locations
    "deosai": (34.9981, 75.3083),
    "deosai national park": (34.9981, 75.3083),
    "shangrila": (35.2971, 75.6333),
    "shangrila resort": (35.2971, 75.6333),
    "katpana": (35.2971, 75.6333),
    "katpana desert": (35.2971, 75.6333),
    "cold desert": (35.2971, 75.6333),
    "skardu fort": (35.2971, 75.6333),
    "khaplu": (35.1581, 76.3333),
    # Hunza area - more specific locations
    "attabad": (36.3167, 74.6500),
    "attabad lake": (36.3167, 74.6500),
    "altit": (36.3167, 74.6500),
    "altit fort": (36.3167, 74.6500),
    "baltit": (36.3167, 74.6500),
    "baltit fort": (36.3167, 74.6500),
    "passu": (36.4833, 74.8167),
    "passu cones": (36.4833, 74.8167),
    "hunza valley": (36.3167, 74.6500),
    # Naran area - more specific locations
    "saif ul muluk": (34.9100, 73.6500),
    "lake saif ul muluk": (34.9100, 73.6500),
    "saif-ul-muluk": (34.9100, 73.6500),
    "lulusar": (34.9100, 73.6500),
    "lulusar lake": (34.9100, 73.6500),
    "babusar": (35.0833, 73.8167),
    "babusar pass": (35.0833, 73.8167),
    "naran valley": (34.9100, 73.6500),
    # Swat area - more specific locations
    "malam jabba": (35.2200, 72.4200),
    "kalam": (35.4833, 72.5833),
    "kalam valley": (35.4833, 72.5833),
    "mahodand": (35.4833, 72.5833),
    "mahodand lake": (35.4833, 72.5833),
    "swat valley": (35.2200, 72.4200),
    "mingora": (34.7797, 72.3600),
    # Murree area - more specific locations
    "patriata": (33.9072, 73.3903),
    "ayubia": (33.9072, 73.3903),
    "ayubia national park": (33.9072, 73.3903),
    "murree hills": (33.9072, 73.3903),
    # Islamabad area
    "faisal mosque": (33.7294, 73.0386),
    "daman-e-koh": (33.6844, 73.0479),
    "margalla hills": (33.6844, 73.0479),
    "lok virsa": (33.6844, 73.0479),
    # Lahore area
    "badshahi mosque": (31.5880, 74.3100),
    "lahore fort": (31.5880, 74.3100),
    "shalimar gardens": (31.5880, 74.3100),
    # Karachi area
    "clifton beach": (24.8000, 67.0300),
    "mazar-e-quaid": (24.8734, 67.0370),
}

def get_destination_image(destination: str) -> Optional[str]:
    """Get destination image URL - returns None if no destination"""
    if not destination or not destination.strip():
        return None
    destination_lower = destination.lower().strip()
    for key in DESTINATION_IMAGES:
        if key in destination_lower:
            return DESTINATION_IMAGES[key]
    return DESTINATION_IMAGES["default"]

def get_destination_coords(destination: str) -> Optional[Tuple[float, float]]:
    """Get destination coordinates"""
    destination_lower = destination.lower()
    for key, coords in DESTINATION_COORDS.items():
        if key in destination_lower:
            return coords
    return None

def get_place_coords(place: str, destination_coords: Optional[Tuple[float, float]] = None) -> Optional[Tuple[float, float]]:
    """Get coordinates for a place, with fallback to destination area"""
    place_lower = place.lower().strip()
    
    # First try exact match in PLACE_COORDS (prioritize longer, more specific matches)
    best_match = None
    best_match_len = 0
    
    for key, coords in PLACE_COORDS.items():
        # Check for exact match first
        if place_lower == key:
            return coords
        # Check if key is contained in place (e.g., "saif ul muluk" contains "saif ul muluk")
        if key in place_lower:
            # Prefer longer, more specific matches
            if len(key) > best_match_len:
                best_match = coords
                best_match_len = len(key)
    
    if best_match:
        return best_match
    
    # Try to find in DESTINATION_COORDS (might be a city name) - exact match only
    for key, coords in DESTINATION_COORDS.items():
        if place_lower == key or key in place_lower:
            # Only use if it's a reasonable match (not too generic)
            if len(key) >= 4:  # Avoid matching very short city names
                return coords
    
    # If destination coords provided, use nearby coordinates with small systematic offset
    # This is better than random - use index-based offset for consistency
    if destination_coords:
        # Use a hash-based offset for consistency (same place = same offset)
        place_hash = hash(place_lower) % 1000
        # Small offset within ~5km
        offset_lat = destination_coords[0] + ((place_hash % 100) - 50) * 0.001
        offset_lon = destination_coords[1] + ((place_hash // 100) % 100 - 50) * 0.001
        return (offset_lat, offset_lon)
    
    return None

def create_destination_map(destination: str, places: Optional[List[str]] = None) -> go.Figure:
    """Create an interactive map for the destination"""
    coords = get_destination_coords(destination)
    
    if not coords:
        # Default to Pakistan center if destination not found
        coords = (30.3753, 69.3451)
    
    lat, lon = coords
    
    fig = go.Figure()
    
    # Add destination marker
    fig.add_trace(go.Scattermap(
        mode="markers+text",
        lat=[lat],
        lon=[lon],
        text=[destination.title()],
        textposition="top center",
        marker=dict(
            size=20,
            color="#0d9488",
        ),
        name="Destination",
        hovertemplate=f"<b>{destination.title()}</b><br>Your destination<extra></extra>"
    ))
    
    # Add places if provided - use actual coordinates when available
    place_coords_list = []
    if places:
        for place in places[:8]:  # Limit to 8 places
            place_coord = get_place_coords(place, coords)
            if place_coord:
                place_coords_list.append((place, place_coord))
        
        # Add markers for places with actual coordinates
        for place, (place_lat, place_lon) in place_coords_list:
            fig.add_trace(go.Scattermap(
                mode="markers+text",
                lat=[place_lat],
                lon=[place_lon],
                text=[place],
                textposition="top center",
                marker=dict(
                    size=15,
                    color="#0891b2",
                ),
                name=place,
                hovertemplate=f"<b>{place}</b><br>📍 Attraction/Place<extra></extra>",
                showlegend=False
            ))
    
    # Adjust zoom based on number of places
    if len(place_coords_list) > 3:
        zoom_level = 9  # Zoom out more if many places
    elif len(place_coords_list) > 0:
        zoom_level = 10
    else:
        zoom_level = 11  # Zoom in more if just destination
    
    fig.update_layout(
        map=dict(
            style="carto-positron",
            center=dict(lat=lat, lon=lon),
            zoom=zoom_level
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248, 250, 252, 0.3)',
        title=dict(
            text=f"📍 {destination.title()} - Map View",
            font=dict(size=18, color="#1e293b", family="Arial, sans-serif", weight="bold"),
            x=0.5,
            xanchor="center"
        )
    )
    
    return fig

def create_budget_chart(budget_data: Dict[str, float], currency: str = "PKR") -> go.Figure:
    """Create a beautiful interactive budget breakdown chart"""
    if not budget_data:
        # Return empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No budget data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="#64748b", family="Arial, sans-serif")
        )
        fig.update_layout(
            title=dict(
                text="💰 Budget Breakdown",
                font=dict(size=24, color="#1e293b", family="Arial, sans-serif"),
                x=0.5,
                xanchor="center"
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#334155", family="Arial, sans-serif"),
            height=400
        )
        return fig
    
    categories = list(budget_data.keys())
    amounts = list(budget_data.values())
    total = sum(amounts)
    
    # Use beautiful gradient colors
    colors = px.colors.sequential.Teal
    
    fig = px.pie(
        values=amounts,
        names=categories,
        color_discrete_sequence=colors,
        hole=0.4  # Donut chart for modern look
    )
    
    # Beautiful hover template with more details
    fig.update_traces(
        textposition='outside',
        textinfo='percent+label',
        texttemplate='<b>%{label}</b><br>%{percent}<br><span style="font-size:10px">%{value:,.0f} ' + currency + '</span>',
        hovertemplate="<b>%{label}</b><br>" +
                      f"Amount: %{{value:,.0f}} {currency}<br>" +
                      "Percentage: %{percent}<br>" +
                      f"Total: {total:,.0f} {currency}<br>" +
                      "<extra></extra>",
        marker=dict(
            line=dict(color='#ffffff', width=2)
        ),
        pull=[0.05] * len(categories)  # Slight pull for depth
    )
    
    # Beautiful layout
    fig.update_layout(
        title=dict(
            text="💰 Budget Breakdown",
            font=dict(size=24, color="#1e293b", family="Arial, sans-serif", weight="bold"),
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top"
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248, 250, 252, 0.5)',
        font=dict(color="#334155", family="Arial, sans-serif", size=12),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.15,
            font=dict(size=12, color="#64748b", family="Arial, sans-serif"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#e2e8f0",
            borderwidth=1
        ),
        height=400,
        margin=dict(l=40, r=200, t=80, b=40),
        annotations=[dict(
            text=f'<b>Total</b><br>{total:,.0f}<br>{currency}',
            x=0.5, y=0.5,
            font_size=16,
            font_color="#1e293b",
            font_family="Arial, sans-serif",
            showarrow=False
        )],
        transition=dict(
            duration=500,
            easing='cubic-in-out'
        )
    )
    
    return fig

def create_timeline_chart(days_data: List[Dict]) -> go.Figure:
    """Create a beautiful animated interactive timeline chart"""
    if not days_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No timeline data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="#64748b")
        )
        fig.update_layout(
            title=dict(
                text="🗓️ Trip Timeline",
                font=dict(size=24, color="#1e293b", family="Arial, sans-serif"),
                x=0.5,
                xanchor="center"
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#334155"),
            height=500,
            margin=dict(l=40, r=40, t=80, b=40)
        )
        return fig
    
    days = []
    activities = []
    dates = []
    places_list = []
    
    # Color gradient for days
    colors = px.colors.sequential.Teal
    
    for day in days_data:
        day_num = day.get('day', '')
        days.append(f"Day {day_num}")
        desc = day.get('description', 'Exploring')
        activities.append(desc)
        dates.append(day.get('date', ''))
        places = day.get('places', [])
        places_list.append(', '.join(places[:3]) if places else '')
    
    num_days = len(days)
    
    # Create a beautiful timeline with connecting lines
    fig = go.Figure()
    
    # Add connecting line
    fig.add_trace(go.Scatter(
        x=list(range(1, num_days + 1)),
        y=[0.5] * num_days,
        mode='lines',
        line=dict(
            color='#0d9488',
            width=4,
            dash='solid'
        ),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Add day markers with gradient colors
    for i, (day, activity, date, places) in enumerate(zip(days, activities, dates, places_list)):
        color_idx = int((i / max(num_days - 1, 1)) * (len(colors) - 1))
        color = colors[color_idx] if color_idx < len(colors) else colors[-1]
        
        # Main day marker
        fig.add_trace(go.Scatter(
            x=[i+1],
            y=[0.5],
            mode='markers+text',
            marker=dict(
                size=50,
                color=color,
                line=dict(width=3, color='white'),
                symbol='circle'
            ),
            text=[str(i+1)],
            textposition="middle center",
            textfont=dict(color="white", size=16, family="Arial, sans-serif", weight="bold"),
            name=day,
            hovertemplate=f"<b>{day}</b>" + 
                         (f"<br>📅 {date}" if date else "") +
                         f"<br>{activity[:100]}{'...' if len(activity) > 100 else ''}" +
                         (f"<br>📍 {places}" if places else "") +
                         "<extra></extra>",
            showlegend=False
        ))
        
        # Add activity text above marker
        activity_short = activity[:40] + "..." if len(activity) > 40 else activity
        fig.add_annotation(
            x=i+1,
            y=0.75,
            text=activity_short,
            showarrow=False,
            font=dict(size=11, color="#334155", family="Arial, sans-serif"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#e2e8f0",
            borderwidth=1,
            borderpad=4,
            xanchor="center"
        )
    
    # Add day labels below
    for i, day in enumerate(days):
        fig.add_annotation(
            x=i+1,
            y=0.25,
            text=day,
            showarrow=False,
            font=dict(size=12, color="#64748b", family="Arial, sans-serif", weight="bold"),
            xanchor="center"
        )
    
    # Beautiful layout with animations
    fig.update_layout(
        title=dict(
            text="🗓️ Trip Timeline",
            font=dict(size=28, color="#1e293b", family="Arial, sans-serif", weight="bold"),
            x=0.5,
            xanchor="center",
            y=0.95,
            yanchor="top"
        ),
        xaxis=dict(
            title=dict(
                text="Days of Your Journey",
                font=dict(size=14, color="#64748b", family="Arial, sans-serif")
            ),
            tickvals=list(range(1, num_days + 1)),
            ticktext=[f"Day {i+1}" for i in range(num_days)],
            showgrid=True,
            gridcolor='rgba(226, 232, 240, 0.5)',
            gridwidth=1,
            zeroline=False,
            range=[0.5, num_days + 0.5]
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[0, 1]
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248, 250, 252, 0.5)',
        font=dict(color="#334155", family="Arial, sans-serif"),
        height=500,
        margin=dict(l=60, r=60, t=100, b=80),
        hovermode='closest',
        # Add smooth transitions
        transition=dict(
            duration=500,
            easing='cubic-in-out'
        )
    )
    
    # Add animation frames for a smooth reveal effect
    frames = []
    for i in range(1, num_days + 1):
        frame_data = []
        # Add line up to current day
        frame_data.append(go.Scatter(
            x=list(range(1, i + 1)),
            y=[0.5] * i,
            mode='lines',
            line=dict(color='#0d9488', width=4),
            showlegend=False,
            hoverinfo='skip'
        ))
        # Add markers up to current day
        for j in range(i):
            color_idx = int((j / max(num_days - 1, 1)) * (len(colors) - 1))
            color = colors[color_idx] if color_idx < len(colors) else colors[-1]
            frame_data.append(go.Scatter(
                x=[j+1],
                y=[0.5],
                mode='markers+text',
                marker=dict(size=50, color=color, line=dict(width=3, color='white')),
                text=[str(j+1)],
                textposition="middle center",
                textfont=dict(color="white", size=16, family="Arial, sans-serif", weight="bold"),
                showlegend=False
            ))
        frames.append(go.Frame(data=frame_data, name=str(i)))
    
    fig.frames = frames
    
    # Add animation controls
    fig.update_layout(
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [{
                'label': '▶️ Play Animation',
                'method': 'animate',
                'args': [None, {
                    'frame': {'duration': 800, 'redraw': True},
                    'fromcurrent': True,
                    'transition': {'duration': 400, 'easing': 'cubic-in-out'}
                }]
            }, {
                'label': '⏸️ Pause',
                'method': 'animate',
                'args': [[None], {
                    'frame': {'duration': 0, 'redraw': False},
                    'mode': 'immediate',
                    'transition': {'duration': 0}
                }]
            }],
            'x': 0.1,
            'y': 0,
            'xanchor': 'left',
            'yanchor': 'bottom',
            'pad': {'t': 10, 'r': 10}
        }]
    )
    
    return fig

def _parse_interests(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    interests = [item.strip() for item in raw.split(",") if item.strip()]
    return interests or None

def format_enhanced_itinerary(response: Dict[str, Any], destination: str = "Pakistan") -> Tuple[str, go.Figure, go.Figure, go.Figure]:
    """Format itinerary with enhanced visual elements including maps"""
    if not response:
        empty_map = create_destination_map(destination)
        return "## No itinerary data available", create_budget_chart({}), create_timeline_chart([]), empty_map
    
    lines = []
    
    # Header with destination image
    destination_city = response.get("destination_city", destination)
    pretty_text = response.get("pretty_itinerary_text")
    
    # If we have pretty_itinerary_text, it contains the full itinerary - display it directly
    if pretty_text:
        # Use the full itinerary text as the main content
        lines.append(f"# 🗺️ {destination_city} Itinerary")
        lines.append("---")
        lines.append("")
        # Display the full itinerary text (day headers should already be plain text)
        lines.append(pretty_text)
        lines.append("")
    else:
        # Fallback if no pretty text
        lines.append(f"# 🗺️ Amazing Trip to {destination_city}")
        lines.append("---")
    
    # Key stats row
    stats_lines = []
    if travel_window := response.get("travel_window"):
        stats_lines.append(f"**📅 When:** {travel_window}")
    if num_days := response.get("num_days"):
        stats_lines.append(f"**⏱️ Duration:** {num_days} days")
    if num_people := response.get("num_of_people"):
        stats_lines.append(f"**👥 Travelers:** {num_people} people")
    
    if stats_lines:
        lines.append(" | ".join(stats_lines))
        lines.append("")
    
    # Budget information
    total_cost = response.get("total_estimated_cost")
    currency = response.get("currency", "PKR")
    budget_data = response.get("budget_breakdown", {})
    
    if total_cost is not None:
        lines.append(f"### 💰 Total Estimated Cost: **{total_cost:,} {currency}**")
        lines.append("")
    
    # Weather considerations
    if weather := response.get("weather_considerations"):
        lines.append(f"### 🌤️ Weather Note")
        lines.append(f"{weather}")
        lines.append("")
    
    # Recommended stays
    if hotels := response.get("recommended_hotels"):
        lines.append("### 🏨 Recommended Stays")
        for hotel in hotels[:3]:  # Show top 3
            lines.append(f"- {hotel}")
        lines.append("")
    
    # Display all images gallery if available
    all_images = response.get("all_images", [])
    if all_images:
        lines.append("### 📸 Destination Gallery")
        lines.append("")
        # Display images in a grid (3 columns)
        image_rows = []
        for i in range(0, len(all_images), 3):
            row_images = all_images[i:i+3]
            image_tags = []
            for img_url in row_images:
                if img_url:
                    image_tags.append(f'<img src="{img_url}" alt="Destination image" style="max-width: 250px; height: auto; margin: 5px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">')
            if image_tags:
                image_rows.append(" ".join(image_tags))
        if image_rows:
            lines.append("\n".join(image_rows))
        lines.append("")
    
    # Collect all places for map from itinerary days
    all_places = []
    days = response.get("days", [])
    if days:
        lines.append("### 📅 Daily Plan")
        for day in days:
            day_num = day.get('day', '')
            date_info = f" · {day['date']}" if day.get('date') else ""
            desc = day.get('description', '')
            places = day.get('places', [])
            
            if places:
                all_places.extend(places)
            
            lines.append(f"#### Day {day_num}{date_info}")
            lines.append(f"{desc}")
            if places:
                lines.append(f"**📍 Highlights:** {', '.join(places)}")
            
            # Display images for this day if available
            day_images = day.get('images', [])
            if day_images:
                lines.append("")
                lines.append("**📸 Images:**")
                # Display up to 4 images per day in a grid
                image_count = min(len(day_images), 4)
                for i in range(0, image_count, 2):
                    row_images = day_images[i:i+2]
                    image_tags = []
                    for img_url in row_images:
                        if img_url:
                            image_tags.append(f'<img src="{img_url}" alt="Day {day_num} image" style="max-width: 300px; height: auto; margin: 5px; border-radius: 8px;">')
                    if image_tags:
                        lines.append(" ".join(image_tags))
                lines.append("")
            else:
                lines.append("")
    
    # Create visualizations with spots from itinerary
    budget_chart = create_budget_chart(budget_data, currency)
    timeline_chart = create_timeline_chart(days)
    # Use destination city and all places from itinerary for map
    destination_map = create_destination_map(destination_city, all_places[:15] if all_places else None)
    
    return "\n".join(lines), budget_chart, timeline_chart, destination_map

def validate_inputs(
    destination_city: str,
    budget_amount: Optional[float],
    num_days: float,
    num_people: float,
) -> Optional[str]:
    """Validate user inputs and return error message if invalid"""
    errors = []
    
    # Validate destination city
    if not destination_city or not destination_city.strip():
        errors.append("❌ Destination city is required")
    elif len(destination_city.strip()) > 100:
        errors.append("❌ Destination city name is too long (max 100 characters)")
    
    # Validate budget
    if budget_amount is None:
        errors.append("❌ Budget amount is required")
    elif budget_amount <= 0:
        errors.append("❌ Budget must be greater than 0")
    elif budget_amount < 10000:
        errors.append("❌ Budget must be at least 10,000 PKR")
    elif budget_amount > 10000000:
        errors.append("❌ Budget cannot exceed 10,000,000 PKR")
    
    # Validate number of days
    if num_days is None or num_days <= 0:
        errors.append("❌ Number of days must be greater than 0")
    elif num_days > 30:
        errors.append("❌ Number of days cannot exceed 30")
    elif not isinstance(num_days, (int, float)) or num_days != int(num_days):
        errors.append("❌ Number of days must be a whole number")
    
    # Validate number of people
    if num_people is None or num_people <= 0:
        errors.append("❌ Number of travelers must be greater than 0")
    elif num_people > 20:
        errors.append("❌ Number of travelers cannot exceed 20")
    elif not isinstance(num_people, (int, float)) or num_people != int(num_people):
        errors.append("❌ Number of travelers must be a whole number")
    
    # Validate budget per person per day
    if budget_amount and num_days and num_people:
        budget_per_person_per_day = budget_amount / (num_days * num_people)
        if budget_per_person_per_day < 500:
            errors.append(f"❌ Budget too low: {budget_per_person_per_day:.0f} PKR per person per day. Minimum 500 PKR required")
    
    if errors:
        return "## ⚠️ Validation Errors\n\n" + "\n".join(f"- {error}" for error in errors)
    
    return None


def call_itinerary_api(
    destination_city: str,
    departure_city: Optional[str],
    region: Optional[str],
    interests_text: Optional[str],
    budget_amount: Optional[float],
    num_days: float,
    transport_mode: Optional[str],
    travel_date: Optional[str],
    num_people: float,
) -> Tuple[str, go.Figure, go.Figure, go.Figure, Dict[str, Any], str, Dict[str, Any]]:
    """Enhanced API call with visualizations including maps"""
    
    # Validate inputs first
    validation_error = validate_inputs(destination_city, budget_amount, num_days, num_people)
    if validation_error:
        empty_map = create_destination_map("Pakistan")
        return validation_error, create_budget_chart({}), create_timeline_chart([]), empty_map, {}, "Validation failed", {}
    
    # Additional validations
    if not destination_city.strip():
        empty_map = create_destination_map("Pakistan")
        return "## Please enter a destination city", create_budget_chart({}), create_timeline_chart([]), empty_map, {}, "Destination city is required.", {}
    
    # Ensure budget_amount is not None after validation
    if budget_amount is None:
        budget_amount = 50000  # Default fallback

    # Convert numeric budget to budget level for backward compatibility
    # But also send the actual budget amount
    if budget_amount < 30000:
        budget_level = "low"
    elif budget_amount > 100000:
        budget_level = "high"
    else:
        budget_level = "medium"
    
    # Validate and clean optional fields
    departure_clean = departure_city.strip() if departure_city and departure_city.strip() else None
    if departure_clean and len(departure_clean) > 100:
        departure_clean = departure_clean[:100]
    
    region_clean = region.strip() if region and region.strip() else None
    if region_clean and len(region_clean) > 100:
        region_clean = region_clean[:100]
    
    travel_date_clean = _normalize_date_input(travel_date)
    
    payload = {
        "destination_city": destination_city.strip()[:100],
        "budget_level": budget_level,
        "budget_amount": float(budget_amount),
        "num_days": int(num_days),
        "num_of_people": int(num_people),
    }
    
    if departure_clean:
        payload["departure_city"] = departure_clean
    if region_clean:
        payload["region"] = region_clean
    if travel_date_clean:
        payload["travel_date"] = travel_date_clean

    if transport_mode and transport_mode.strip():
        # Extract transport mode (remove emoji if present)
        transport_clean = transport_mode.replace("🚗", "").replace("🚌", "").replace("🚕", "").replace("🔄", "").strip().lower()
        # Map to backend expected values
        if "own car" in transport_clean or "car" in transport_clean:
            payload["transport_mode"] = "own_car"
        elif "public" in transport_clean:
            payload["transport_mode"] = "public_transport"
        elif "ride" in transport_clean or "sharing" in transport_clean:
            payload["transport_mode"] = "ride_sharing"
        elif "mixed" in transport_clean:
            payload["transport_mode"] = "mixed"
        else:
            payload["transport_mode"] = transport_clean
    if interests := _parse_interests(interests_text):
        payload["interests"] = interests

    url = f"{API_BASE}{ITINERARY_ENDPOINT}"

    try:
        response = requests.post(url, json=payload, timeout=90)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        error_msg = f"## ❌ Request Failed\n\nError: {exc}\n\nPlease check if the backend server is running."
        empty_map = create_destination_map("Pakistan")
        return error_msg, create_budget_chart({}), create_timeline_chart([]), empty_map, {}, f"Request failed: {exc}", {}
    except ValueError:
        error_msg = "## ❌ Invalid Response\n\nThe backend returned a non-JSON response."
        empty_map = create_destination_map("Pakistan")
        return error_msg, create_budget_chart({}), create_timeline_chart([]), empty_map, {}, "Backend returned a non-JSON response.", {}

    markdown, budget_chart, timeline_chart, destination_map = format_enhanced_itinerary(data, destination_city.strip())
    status = "✅ Itinerary generated successfully!" if markdown else "⚠️ Received response from API."
    
    return markdown, budget_chart, timeline_chart, destination_map, data, status, data

def _serialize_history(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Convert messages format history to API format (same format)"""
    return history

def chat_with_itinerary(
    message: str, history: List[Dict[str, str]], itinerary_context: Optional[Dict[str, Any]]
) -> Tuple[List[Dict[str, str]], str, Optional[Dict[str, Any]]]:
    """Chat with itinerary assistant, returns history, empty string, and optionally updated itinerary"""
    user_message = (message or "").strip()
    if not user_message:
        return history, "", None
    
    # Validate message length
    if len(user_message) > 2000:
        error_reply = "❌ Message is too long. Please keep it under 2000 characters."
        updated_history = history + [
            {"role": "user", "content": user_message[:2000] + "..."},
            {"role": "assistant", "content": error_reply}
        ]
        return updated_history, "", None
    
    # Validate history length
    if len(history) > 50:
        # Keep only last 50 messages
        history = history[-50:]

    payload = {
        "message": user_message,
        "history": _serialize_history(history),
    }
    if itinerary_context:
        payload["itinerary_context"] = itinerary_context

    url = f"{API_BASE}{CHATBOT_ENDPOINT}"

    updated_itinerary = None
    try:
        response = requests.post(url, json=payload, timeout=90)
        response.raise_for_status()
        data = response.json()
        reply = data.get("reply", "I couldn't understand the response from the chatbot.")
        
        # Check if the API returned an updated itinerary
        if "updated_itinerary" in data:
            updated_itinerary = data["updated_itinerary"]
        elif "itinerary" in data:
            updated_itinerary = data["itinerary"]
    except requests.RequestException as exc:
        reply = f"❌ Chatbot request failed: {exc}"
    except ValueError:
        reply = "❌ Chatbot returned a non-JSON response."

    # Add user message and assistant reply in messages format
    updated_history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": reply}
    ]
    return updated_history, "", updated_itinerary

# User authentication functions
def register_user(email: str, password: str, full_name: str) -> Tuple[str, str]:
    """Register a new user."""
    if not email or not password:
        return "❌ Email and password are required", ""
    
    try:
        payload = {
            "email": email.strip(),
            "password": password,
        }
        if full_name and full_name.strip():
            payload["full_name"] = full_name.strip()
        
        response = requests.post(
            f"{API_BASE}{USER_REGISTER_ENDPOINT}",
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        token = data.get("token", "")
        user_info = data.get("user", {})
        return f"✅ Registration successful! Welcome, {user_info.get('full_name', user_info.get('email', 'User'))}!", token
    except requests.RequestException as e:
        error_msg = str(e)
        if "400" in error_msg or "already registered" in error_msg.lower():
            return "❌ Email already registered. Please login instead.", ""
        return f"❌ Registration failed: {error_msg}", ""
    except Exception as e:
        return f"❌ Error: {str(e)}", ""

def login_user(email: str, password: str) -> Tuple[str, str]:
    """Login user."""
    if not email or not password:
        return "❌ Email and password are required", ""
    
    try:
        payload = {
            "email": email.strip(),
            "password": password
        }
        response = requests.post(
            f"{API_BASE}{USER_LOGIN_ENDPOINT}",
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        token = data.get("token", "")
        user_info = data.get("user", {})
        return f"✅ Login successful! Welcome back, {user_info.get('full_name', user_info.get('email', 'User'))}!", token
    except requests.RequestException as e:
        error_msg = str(e)
        if "400" in error_msg or "invalid" in error_msg.lower():
            return "❌ Invalid email or password.", ""
        return f"❌ Login failed: {error_msg}", ""
    except Exception as e:
        return f"❌ Error: {str(e)}", ""

def get_user_profile(token: str) -> str:
    """Get user profile."""
    if not token:
        return "Not logged in"
    
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(
            f"{API_BASE}{USER_PROFILE_ENDPOINT}",
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        name = data.get("full_name", "")
        email = data.get("email", "")
        if name:
            return f"👤 {name} ({email})"
        return f"👤 {email}"
    except:
        return "Not logged in"



# ─────────────────────────────────────────────────────────────────────────────
# PLAN YOUR TRIP — API helpers
# ─────────────────────────────────────────────────────────────────────────────

TRIP_OPTIONS_ENDPOINT = "/api/trip/options"
TRIP_ITINERARY_ENDPOINT = "/api/trip/itinerary"

WEATHER_CODE_LABELS = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
}


def _normalize_date_input(value: Any) -> Optional[str]:
    """Normalize date-like inputs to YYYY-MM-DD string."""
    if value is None:
        return None

    if isinstance(value, datetime):
        return value.date().isoformat()

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if "T" in raw:
            raw = raw.split("T", 1)[0]
        return raw[:50]

    return str(value)[:50]


def build_transport_redirect_markdown(
    mode: Optional[str],
    origin: Optional[str],
    destination: Optional[str],
    travel_date: Optional[Any],
) -> str:
    """Build a transport provider redirect link based on selected mode."""
    mode_clean = (mode or "").strip().lower()
    date_value = _normalize_date_input(travel_date) or datetime.now().date().isoformat()
    origin_value = (origin or "").strip() or "Lahore"
    destination_value = (destination or "").strip() or "Multan"

    if mode_clean == "bus":
        bus_params = urlencode(
            {"from": origin_value, "to": destination_value, "date": date_value}
        )
        url = f"https://bookkaru.com/bus/listing?{bus_params}"
        return (
            "### 🚌 Bus Booking Redirect\n\n"
            f"[Open BookKaru Bus Listing]({url})\n\n"
            f"`{url}`"
        )
    if mode_clean == "indrive":
        url = "https://indrive.com/"
        return f"### 🚕 InDrive Redirect\n\n[Open InDrive]({url})\n\n`{url}`"
    if mode_clean == "train":
        url = "https://www.pakrailways.gov.pk/train"
        return f"### 🚂 Train Booking Redirect\n\n[Open Pakistan Railways]({url})\n\n`{url}`"

    return "❌ Please choose a transport mode first (Bus, InDrive, or Train)."


def build_daily_weather_markdown(
    destination: Optional[str],
    travel_date: Optional[Any],
    num_days: int,
) -> str:
    """Fetch and format daily weather for selected travel days."""
    destination_clean = (destination or "").strip()
    if not destination_clean:
        return "❌ Enter a destination city to load weather."

    safe_days = max(1, min(int(num_days or 1), 14))
    start_date_str = _normalize_date_input(travel_date) or datetime.now().date().isoformat()
    try:
        start_date = datetime.fromisoformat(start_date_str).date()
    except ValueError:
        start_date = datetime.now().date()

    end_date = start_date + timedelta(days=safe_days - 1)

    try:
        geo_resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": destination_clean, "count": 1, "language": "en", "format": "json"},
            timeout=15,
        )
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()
        results = geo_data.get("results", [])
        if not results:
            return f"❌ Could not find location for `{destination_clean}`."

        first = results[0]
        lat = first.get("latitude")
        lon = first.get("longitude")
        resolved_name = first.get("name", destination_clean)
        country = first.get("country", "")

        weather_resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "weathercode,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "timezone": "auto",
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
            timeout=15,
        )
        weather_resp.raise_for_status()
        weather_data = weather_resp.json().get("daily", {})
    except requests.RequestException as exc:
        return f"❌ Weather service request failed: {exc}"
    except (ValueError, TypeError):
        return "❌ Weather service returned an invalid response."

    dates = weather_data.get("time", [])
    t_max = weather_data.get("temperature_2m_max", [])
    t_min = weather_data.get("temperature_2m_min", [])
    weather_codes = weather_data.get("weathercode", [])
    precip = weather_data.get("precipitation_probability_max", [])

    if not dates:
        return "❌ No weather data available for the selected dates."

    location_label = f"{resolved_name}, {country}".strip(", ")
    lines = [f"### 🌤️ Daily Weather Forecast ({location_label})", ""]
    lines.append("| Date | Weather | Min / Max (°C) | Rain Chance |")
    lines.append("|---|---|---|---|")

    for i, day_str in enumerate(dates):
        code = weather_codes[i] if i < len(weather_codes) else None
        weather_text = WEATHER_CODE_LABELS.get(code, "Unknown")
        min_c = t_min[i] if i < len(t_min) else "N/A"
        max_c = t_max[i] if i < len(t_max) else "N/A"
        rain = precip[i] if i < len(precip) else "N/A"
        lines.append(f"| {day_str} | {weather_text} | {min_c} / {max_c} | {rain}% |")

    return "\n".join(lines)


def call_trip_options(
    origin: str,
    destination: str,
    num_days: int,
    num_travelers: int,
) -> Tuple[str, List[str], List[str], Any]:
    """
    Calls POST /api/trip/options.
    Returns: (status_msg, hotel_choices, transport_choices, raw_response)
    hotel_choices / transport_choices are human-readable option strings for gr.Radio.
    """
    if not origin.strip():
        return "❌ Origin city is required.", [], [], None
    if not destination.strip():
        return "❌ Destination city is required.", [], [], None
    if num_days < 1 or num_days > 30:
        return "❌ Number of days must be between 1 and 30.", [], [], None
    if num_travelers < 1 or num_travelers > 20:
        return "❌ Number of travelers must be between 1 and 20.", [], [], None

    payload = {
        "origin": origin.strip(),
        "destination": destination.strip(),
        "num_days": int(num_days),
        "num_travelers": int(num_travelers),
    }

    try:
        resp = requests.post(
            f"{API_BASE}{TRIP_OPTIONS_ENDPOINT}", json=payload, timeout=60
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        return f"❌ Request failed: {exc}", [], [], None
    except ValueError:
        return "❌ Backend returned a non-JSON response.", [], [], None

    hotels = data.get("hotel_options", [])
    transports = data.get("transport_options", [])
    dist = data.get("distance_km", 0)

    hotel_choices = []
    for i, h in enumerate(hotels):
        stars = "⭐" * int(h.get("star_rating", 3))
        name = h.get("name", f"Hotel {i+1}")
        location = h.get("location", "")
        ppn = h.get("price_per_night", 0)
        total = h.get("total_price", ppn * num_days)
        label = f"{stars} {name} — {location} | PKR {ppn:,.0f}/night (Total: PKR {total:,.0f})"
        hotel_choices.append(label)

    transport_choices = []
    for i, t in enumerate(transports):
        t_type = t.get("type", "").capitalize()
        provider = t.get("provider", "")
        duration = t.get("duration", "")
        total = t.get("total_price", 0)
        ppp = t.get("price_per_person", 0)
        label = f"{_transport_emoji(t_type)} {provider} | PKR {ppp:,.0f}/person · Total PKR {total:,.0f} · ⏱ {duration}"
        transport_choices.append(label)

    status = f"✅ Options loaded! Route: {origin} → {destination} ({dist:.0f} km). Select a hotel and transport below."
    return status, hotel_choices, transport_choices, data


def _transport_emoji(t_type: str) -> str:
    mapping = {"Flight": "✈️", "Bus": "🚌", "Train": "🚂", "Car": "🚗"}
    for key, emoji in mapping.items():
        if key.lower() in t_type.lower():
            return emoji
    return "🚐"


def call_generate_trip_itinerary(
    options_data: Any,
    hotel_idx: int,
    transport_idx: int,
) -> str:
    """
    Calls POST /api/trip/itinerary with SelectionRequest.
    Returns formatted markdown string.
    """
    if not options_data:
        return "❌ Please fetch trip options first."
    if hotel_idx < 0:
        return "❌ Please select a hotel."
    if transport_idx < 0:
        return "❌ Please select a transport option."

    trip_request = options_data.get("_trip_request", {})
    hotel_options = options_data.get("hotel_options", [])
    transport_options = options_data.get("transport_options", [])

    if hotel_idx >= len(hotel_options):
        return "❌ Hotel selection out of range. Please fetch options again."
    if transport_idx >= len(transport_options):
        return "❌ Transport selection out of range. Please fetch options again."

    payload = {
        "trip_request": trip_request,
        "selected_hotel_index": hotel_idx,
        "selected_transport_index": transport_idx,
        "hotel_options": hotel_options,
        "transport_options": transport_options,
    }

    try:
        resp = requests.post(
            f"{API_BASE}{TRIP_ITINERARY_ENDPOINT}", json=payload, timeout=90
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        return f"❌ Request failed: {exc}"
    except ValueError:
        return "❌ Backend returned a non-JSON response."

    return _format_trip_itinerary(data, trip_request)


def _format_trip_itinerary(data: Dict[str, Any], trip_request: Optional[Dict[str, Any]] = None) -> str:
    """Format TripItineraryResponse into a clean markdown string."""
    lines = []
    cs = data.get("cost_summary", {})

    hotel_name = cs.get("hotel_name", "Selected Hotel")
    hotel_ppn = cs.get("hotel_price_per_night", 0)
    hotel_total = cs.get("hotel_total", 0)
    transport_name = cs.get("transport_name", "Selected Transport")
    transport_total = cs.get("transport_total", 0)
    grand_total = cs.get("total_estimated_cost", 0)
    note = cs.get("note", "Activities and food are not included in this estimate")

    lines.append("## 💰 Estimated Cost Summary")
    lines.append("")
    if trip_request:
        origin = trip_request.get("origin", "")
        destination = trip_request.get("destination", "")
        td_days = int(trip_request.get("num_days", 0) or 0)
        td_trav = int(trip_request.get("num_travelers", 1) or 1)
        lines.append("### 🧳 Trip Details")
        lines.append("")
        lines.append("| | |")
        lines.append("|---|---|")
        if origin or destination:
            lines.append(f"| 📍 **Route** | {origin} → {destination} |")
        if td_days:
            lines.append(f"| 📅 **Duration** | {td_days} day(s) |")
        lines.append(f"| 👥 **Travelers** | {td_trav} person(s) |")
        lines.append("")
    lines.append("### 💵 Cost Breakdown")
    lines.append("")
    lines.append("| | |")
    lines.append("|---|---|")
    lines.append(f"| 🏨 **Hotel** | {hotel_name} |")
    if hotel_ppn > 0 and trip_request:
        num_days = int(trip_request.get("num_days", 0) or 0)
        num_travelers = int(trip_request.get("num_travelers", 1) or 1)
        num_rooms = max(1, -(-num_travelers // 2))  # ceil division
        lines.append(
            f"| | PKR {hotel_ppn:,.0f}/room/night × {num_days} night(s) × {num_rooms} room(s) = **PKR {hotel_total:,.0f}** |"
        )
    elif hotel_ppn > 0:
        ratio = round(hotel_total / hotel_ppn) if hotel_ppn else 0
        lines.append(f"| | PKR {hotel_ppn:,.0f}/room/night × {ratio} (nights × rooms) = **PKR {hotel_total:,.0f}** |")
    else:
        lines.append(f"| | **PKR {hotel_total:,.0f}** |")
    lines.append(f"| 🚌 **Transport** | {transport_name} |")
    if trip_request:
        num_travelers = int(trip_request.get("num_travelers", 1) or 1)
        t_type = (cs.get("transport_name") or "").lower()
        if "car" in t_type:
            num_cars = max(1, -(-num_travelers // 4))  # ceil division
            per_car = transport_total / num_cars if num_cars else transport_total
            lines.append(
                f"| | PKR {per_car:,.0f}/vehicle × {num_cars} vehicle(s) = **PKR {transport_total:,.0f}** |"
            )
        else:
            per_person = transport_total / num_travelers if num_travelers else transport_total
            lines.append(
                f"| | PKR {per_person:,.0f}/person × {num_travelers} traveler(s) = **PKR {transport_total:,.0f}** |"
            )
    else:
        lines.append(f"| | Total: **PKR {transport_total:,.0f}** |")
    lines.append(f"| 🧾 **TOTAL ESTIMATED COST** | **PKR {grand_total:,.0f}** |")
    lines.append("")
    lines.append(f"> ⚠️ {note}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 🗓️ Day-by-Day Itinerary")
    lines.append("")

    days = data.get("days", [])
    for day in days:
        day_num = day.get("day_number", "")
        title = day.get("title", f"Day {day_num}")
        hotel = day.get("hotel", hotel_name)
        activities = day.get("activities", [])

        lines.append(f"### {title}")
        lines.append(f"🏨 *Staying at: {hotel}*")
        lines.append("")
        for act in activities:
            lines.append(f"- {act}")
        lines.append("")

    return "\n".join(lines)


def build_interface() -> gr.Blocks:

    with gr.Blocks(
        title="Smart itinerary planner • Professional Itinerary Generator",
        theme=THEME,
        css=CUSTOM_CSS
    ) as demo:
        
        # Authentication Section
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 🔐 Account")
                    with gr.Tabs():
                        with gr.TabItem("Login"):
                            login_email = gr.Textbox(
                                label="Email",
                                placeholder="your@email.com",
                                container=False
                            )
                            login_password = gr.Textbox(
                                label="Password",
                                type="password",
                                container=False
                            )
                            login_btn = gr.Button("Login", variant="primary")
                            login_status = gr.Textbox(
                                label="Status",
                                interactive=False,
                                container=False
                            )
                        
                        with gr.TabItem("Sign Up"):
                            register_email = gr.Textbox(
                                label="Email",
                                placeholder="your@email.com",
                                container=False
                            )
                            register_password = gr.Textbox(
                                label="Password",
                                type="password",
                                container=False
                            )
                            register_name = gr.Textbox(
                                label="Full Name (Optional)",
                                placeholder="Your Name",
                                container=False
                            )
                            register_btn = gr.Button("Sign Up", variant="primary")
                            register_status = gr.Textbox(
                                label="Status",
                                interactive=False,
                                container=False
                            )
                    
                    user_info = gr.Textbox(
                        label="Logged in as",
                        value="Not logged in",
                        interactive=False,
                        container=False
                    )
        
        # Hero Section
        with gr.Row():
            with gr.Column():
                gr.HTML(
                    """
                    <div id="hero-section">
                        <div class="hero-content">
                            <h1 class="hero-title">Smart itinerary planner</h1>
                            <p class="hero-subtitle">Craft extraordinary journeys with AI-powered precision. Personalized itineraries, real-time insights, and seamless planning.</p>
                            <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 2rem;">
                                <div class="stats-card">
                                    <span class="stats-number">5,000+</span>
                                    <span class="stats-label">Trips Planned</span>
                                </div>
                                <div class="stats-card">
                                    <span class="stats-number">98%</span>
                                    <span class="stats-label">Satisfaction</span>
                                </div>
                                <div class="stats-card">
                                    <span class="stats-number">50+</span>
                                    <span class="stats-label">Destinations</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    """
                )

        with gr.Row():
            # Input Section
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 🎯 Trip Details")
                    
                    destination = gr.Textbox(
                        label="🏁 Destination City *",
                        placeholder="e.g., Skardu, Hunza, Naran, Swat...",
                        info="Where do you want to go?",
                        container=False
                    )
                    
                    departure = gr.Textbox(
                        label="📍 Departure City",
                        placeholder="e.g., Islamabad, Karachi, Lahore...",
                        container=False
                    )
                    
                    region = gr.Textbox(
                        label="🗺️ Region / Province",
                        placeholder="e.g., Gilgit-Baltistan, KPK...",
                        container=False
                    )
                    
                    interests = gr.Textbox(
                        label="🎨 Interests & Activities",
                        placeholder="hiking, photography, local food, culture, shopping...",
                        info="Separate with commas",
                        container=False
                    )

                with gr.Group():
                    gr.Markdown("### ⚙️ Trip Configuration")
                    
                    budget = gr.Number(
                        label="💰 Budget (PKR)",
                        value=50000,
                        minimum=10000,
                        maximum=1000000,
                        step=5000,
                        info="Total budget in Pakistani Rupees for the entire trip",
                        container=False
                    )
                    
                    num_days = gr.Slider(
                        1, 21, value=4, step=1,
                        label="📅 Number of Days",
                        info="How long is your trip?",
                        container=False
                    )
                    
                    num_people = gr.Slider(
                        1, 10, value=2, step=1,
                        label="👥 Number of Travelers",
                        info="How many people are traveling?",
                        container=False
                    )

                with gr.Accordion("🎛️ Advanced Options", open=False):
                    transport = gr.Dropdown(
                        choices=["🚗 Own Car", "🚌 Public Transport", "🚕 Ride Sharing", "🔄 Mixed"],
                        label="Primary Transport Mode",
                        value="🚗 Own Car",
                        container=False
                    )
                    
                    travel_date = gr.DateTime(
                        label="📅 Travel Date",
                        info="Select your travel date from the calendar",
                    )

                submit_btn = gr.Button(
                    "🚀 Generate Smart Itinerary",
                    variant="primary",
                    size="lg",
                    elem_classes="cta-button"
                )

            # Output Section
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("🎯 Itinerary Overview"):
                        status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            container=False
                        )
                        
                        with gr.Row():
                            with gr.Column(scale=2):
                                itinerary_md = gr.Markdown(
                                    "## Your itinerary will appear here\n\nConfigure your trip details and click 'Generate Smart Itinerary' to get started!",
                                    elem_id="itinerary-md"
                                )
                    
                    with gr.TabItem("🗺️ Destination Map"):
                        gr.Markdown("### Interactive Map of Your Destination\n\nExplore your destination and key places on this interactive map.")
                        destination_map = gr.Plot(
                            label="Destination Map",
                            show_label=False,
                            container=False,
                            value=create_destination_map("Pakistan")
                        )
                    
                    with gr.TabItem("📊 Visual Analytics"):
                        gr.Markdown(
                            "### 📈 Visual Analytics Dashboard\n"
                            "Explore your trip budget and timeline with beautiful interactive visualizations."
                        )
                        
                        # Timeline - Full width and prominent
                        gr.Markdown("#### 🗓️ Trip Timeline")
                        timeline_chart = gr.Plot(
                            label="Trip Timeline", 
                            show_label=False,
                            container=False
                        )
                        
                        # Budget chart below timeline
                        gr.Markdown("#### 💰 Budget Breakdown")
                        with gr.Row():
                            budget_chart = gr.Plot(
                                label="Budget Breakdown",
                                show_label=False,
                                container=False
                            )
                    
                    with gr.TabItem("🤖 Travel Assistant"):
                        gr.Markdown(
                            "### 💬 Chat with your Travel Assistant\n"
                            "Ask questions about your itinerary, request modifications, or get travel tips!"
                        )
                        
                        chatbot = gr.Chatbot(
                            label="Travel Assistant",
                            height=400,
                            show_copy_button=True,
                            container=False,
                            type="messages"
                        )
                        
                        with gr.Row():
                            chat_input = gr.Textbox(
                                label="Your message",
                                placeholder="e.g., Can we add more hiking activities? What's the best time to visit?",
                                lines=2,
                                scale=4,
                                container=False
                            )
                            chat_send = gr.Button(
                                "Send",
                                variant="primary",
                                scale=1,
                                min_width=100
                            )
                        
                        with gr.Row():
                            chat_clear = gr.Button("Clear Chat", variant="secondary")
                    
                    with gr.TabItem("📋 Raw Data"):
                        raw_json = gr.JSON(
                            label="API Response Data",
                            container=False
                        )
                    
                    with gr.TabItem("✈️ Plan Your Trip"):
                        gr.Markdown(
                            "### ✈️ Plan Your Trip\n"
                            "Enter your destination, choose a hotel and transport — we'll build your itinerary!\n"
                            "> No budget input needed. Options range from budget to luxury automatically."
                        )
                        
                        with gr.Group():
                            gr.Markdown("#### 🗺️ Where & When")
                            with gr.Row():
                                trip_origin = gr.Textbox(
                                    label="🏠 Origin City *",
                                    placeholder="e.g., Islamabad, Lahore, Karachi",
                                    container=False,
                                )
                                trip_destination = gr.Textbox(
                                    label="📍 Destination City *",
                                    placeholder="e.g., Lahore, Swat, Naran, Murree",
                                    container=False,
                                )
                            with gr.Row():
                                trip_days = gr.Slider(
                                    1, 21, value=3, step=1,
                                    label="📅 Number of Days",
                                    container=False,
                                )
                                trip_travelers = gr.Slider(
                                    1, 10, value=1, step=1,
                                    label="👥 Number of Travelers",
                                    container=False,
                                )
                                trip_start_date = gr.DateTime(
                                    label="🗓️ Start Date",
                                    info="Choose start date from calendar",
                                )
                        
                        trip_options_btn = gr.Button(
                            "🔍 Get Hotel & Transport Options",
                            variant="primary",
                            size="lg",
                        )
                        trip_options_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            container=False,
                        )
                        
                        with gr.Group():
                            gr.Markdown("#### 🌤️ Daily Weather")
                            trip_weather_md = gr.Markdown(
                                "Select destination and date, then click **Get Hotel & Transport Options**."
                            )

                        with gr.Group():
                            gr.Markdown("#### 🔗 Transport Redirect")
                            trip_redirect_mode = gr.Dropdown(
                                choices=["Bus", "InDrive", "Train"],
                                label="Choose Transport Provider",
                                value="Bus",
                                container=False,
                            )
                            trip_redirect_btn = gr.Button(
                                "Open Provider Link",
                                variant="secondary",
                            )
                            trip_redirect_md = gr.Markdown(
                                "Choose mode, origin, destination, and date to generate provider link."
                            )

                        with gr.Group():
                            gr.Markdown("#### 🏨 Select a Hotel")
                            trip_hotel_radio = gr.Radio(
                                choices=[],
                                label="Available Hotels",
                                container=False,
                            )
                        
                        with gr.Group():
                            gr.Markdown("#### 🚌 Select Transport")
                            trip_transport_radio = gr.Radio(
                                choices=[],
                                label="Available Transport Options",
                                container=False,
                            )
                        
                        trip_generate_btn = gr.Button(
                            "🗺️ Generate Itinerary",
                            variant="primary",
                            size="lg",
                        )
                        
                        trip_itinerary_md = gr.Markdown(
                            "*Your cost summary and itinerary will appear here after generating.*"
                        )

        # State management
        itinerary_state = gr.State({})
        auth_token = gr.State("")  # Store authentication token
        trip_options_state = gr.State(None)  # Store /api/trip/options response

        
        # Authentication event handlers
        def handle_login(email, password, token):
            """Handle login and update token."""
            status_msg, new_token = login_user(email, password)
            if new_token:
                profile = get_user_profile(new_token)
                return status_msg, new_token, profile
            return status_msg, token, "Not logged in"
        
        def handle_register(email, password, full_name, token):
            """Handle registration and update token."""
            status_msg, new_token = register_user(email, password, full_name)
            if new_token:
                profile = get_user_profile(new_token)
                return status_msg, new_token, profile
            return status_msg, token, "Not logged in"
        
        login_btn.click(
            fn=handle_login,
            inputs=[login_email, login_password, auth_token],
            outputs=[login_status, auth_token, user_info]
        )
        
        register_btn.click(
            fn=handle_register,
            inputs=[register_email, register_password, register_name, auth_token],
            outputs=[register_status, auth_token, user_info]
        )

        # Event handlers
        submit_btn.click(
            fn=call_itinerary_api,
            inputs=[
                destination, departure, region, interests, budget,
                num_days, transport, travel_date, num_people
            ],
            outputs=[itinerary_md, budget_chart, timeline_chart, destination_map, itinerary_state, status, raw_json]
        )

        # Helper function to handle chat and itinerary updates
        def handle_chat_with_update(message, history, context):
            """Handle chat and update itinerary if provided"""
            updated_history, _, updated_itinerary = chat_with_itinerary(message, history, context)
            
            if updated_itinerary:
                # Format and return updated itinerary
                dest = updated_itinerary.get("destination_city", "Pakistan")
                markdown, budget_chart, timeline_chart, destination_map = format_enhanced_itinerary(updated_itinerary, dest)
                return (
                    updated_history,  # chatbot
                    "",  # chat_input
                    markdown,  # itinerary_md
                    budget_chart,  # budget_chart
                    timeline_chart,  # timeline_chart
                    destination_map,  # destination_map
                    updated_itinerary,  # itinerary_state
                    "✅ Itinerary updated from chat!",  # status
                    updated_itinerary  # raw_json
                )
            else:
                # No itinerary update, return None for itinerary outputs
                return (
                    updated_history,  # chatbot
                    "",  # chat_input
                    gr.update(),  # itinerary_md (no update)
                    gr.update(),  # budget_chart (no update)
                    gr.update(),  # timeline_chart (no update)
                    gr.update(),  # destination_map (no update)
                    context,  # itinerary_state (keep existing)
                    gr.update(),  # status (no update)
                    gr.update()  # raw_json (no update)
                )

        # Chat functionality with itinerary update
        chat_send.click(
            fn=handle_chat_with_update,
            inputs=[chat_input, chatbot, itinerary_state],
            outputs=[chatbot, chat_input, itinerary_md, budget_chart, timeline_chart, destination_map, itinerary_state, status, raw_json]
        ).then(
            lambda: gr.update(interactive=True),
            outputs=[chat_input]
        )

        chat_clear.click(
            lambda: ([], ""),
            outputs=[chatbot, chat_input],
            queue=False
        )

        # Enter key for chat
        chat_input.submit(
            fn=handle_chat_with_update,
            inputs=[chat_input, chatbot, itinerary_state],
            outputs=[chatbot, chat_input, itinerary_md, budget_chart, timeline_chart, destination_map, itinerary_state, status, raw_json]
        ).then(
            lambda: gr.update(interactive=True),
            outputs=[chat_input]
        )


        # ── Plan Your Trip event handlers ─────────────────────────────────────

        def handle_trip_options(origin, destination, num_days, num_travelers, start_date):
            """Get hotel & transport options and populate radio choices."""
            status_msg, hotel_choices, transport_choices, data = call_trip_options(
                origin, destination, int(num_days), int(num_travelers)
            )
            weather_md = build_daily_weather_markdown(destination, start_date, int(num_days))
            if data is not None:
                # Embed the trip_request into the options data so itinerary call can use it
                data["_trip_request"] = {
                    "origin": origin.strip(),
                    "destination": destination.strip(),
                    "num_days": int(num_days),
                    "num_travelers": int(num_travelers),
                }
            return (
                status_msg,
                gr.update(choices=hotel_choices, value=None),
                gr.update(choices=transport_choices, value=None),
                data,
                weather_md,
            )

        def handle_transport_redirect(mode, origin, destination, start_date):
            """Build provider redirect link based on selected transport mode."""
            return build_transport_redirect_markdown(mode, origin, destination, start_date)

        def handle_trip_generate(options_data, hotel_choice, transport_choice):
            """Map selected radio labels back to indices and generate itinerary."""
            if not options_data:
                return "❌ Please click 'Get Hotel & Transport Options' first."
            if not hotel_choice:
                return "❌ Please select a hotel."
            if not transport_choice:
                return "❌ Please select a transport option."

            hotels = options_data.get("hotel_options", [])
            transports = options_data.get("transport_options", [])

            # Build the same labels as handle_trip_options to find the matching index
            num_days = options_data.get("_trip_request", {}).get("num_days", 1)
            num_travelers = options_data.get("_trip_request", {}).get("num_travelers", 1)

            hotel_idx = -1
            for i, h in enumerate(hotels):
                stars = "⭐" * int(h.get("star_rating", 3))
                name = h.get("name", f"Hotel {i+1}")
                location = h.get("location", "")
                ppn = h.get("price_per_night", 0)
                total = h.get("total_price", ppn * num_days)
                label = f"{stars} {name} — {location} | PKR {ppn:,.0f}/night (Total: PKR {total:,.0f})"
                if label == hotel_choice:
                    hotel_idx = i
                    break

            transport_idx = -1
            for i, t in enumerate(transports):
                t_type = t.get("type", "").capitalize()
                provider = t.get("provider", "")
                duration = t.get("duration", "")
                total = t.get("total_price", 0)
                ppp = t.get("price_per_person", 0)
                label = f"{_transport_emoji(t_type)} {provider} | PKR {ppp:,.0f}/person · Total PKR {total:,.0f} · ⏱ {duration}"
                if label == transport_choice:
                    transport_idx = i
                    break

            return call_generate_trip_itinerary(options_data, hotel_idx, transport_idx)

        trip_options_btn.click(
            fn=handle_trip_options,
            inputs=[trip_origin, trip_destination, trip_days, trip_travelers, trip_start_date],
            outputs=[trip_options_status, trip_hotel_radio, trip_transport_radio, trip_options_state, trip_weather_md],
        )

        trip_redirect_btn.click(
            fn=handle_transport_redirect,
            inputs=[trip_redirect_mode, trip_origin, trip_destination, trip_start_date],
            outputs=[trip_redirect_md],
        )
        trip_redirect_mode.change(
            fn=handle_transport_redirect,
            inputs=[trip_redirect_mode, trip_origin, trip_destination, trip_start_date],
            outputs=[trip_redirect_md],
        )
        trip_origin.change(
            fn=handle_transport_redirect,
            inputs=[trip_redirect_mode, trip_origin, trip_destination, trip_start_date],
            outputs=[trip_redirect_md],
        )
        trip_destination.change(
            fn=handle_transport_redirect,
            inputs=[trip_redirect_mode, trip_origin, trip_destination, trip_start_date],
            outputs=[trip_redirect_md],
        )
        trip_start_date.change(
            fn=handle_transport_redirect,
            inputs=[trip_redirect_mode, trip_origin, trip_destination, trip_start_date],
            outputs=[trip_redirect_md],
        )

        trip_generate_btn.click(
            fn=handle_trip_generate,
            inputs=[trip_options_state, trip_hotel_radio, trip_transport_radio],
            outputs=[trip_itinerary_md],
        )

    return demo


if __name__ == "__main__":
    app = build_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )