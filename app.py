
import streamlit as st
import pandas as pd
import json
import requests
import os

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

st.set_page_config(
    page_title="AI Travel Planning Assistant",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ AI-Based Travel Planning Assistant")
st.write(
    "Plan your trips with Agentic AI using LangChain and Gemini."
)

st.success("Streamlit application started successfully!")

# Load datasets
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "flights.json"), "r") as file:
    flight_data = json.load(file)

with open(os.path.join(BASE_DIR, "hotels.json"), "r") as file:
    hotel_data = json.load(file)

with open(os.path.join(BASE_DIR, "places.json"), "r") as file:
    places_data = json.load(file)

flight_df = pd.DataFrame(flight_data)
hotel_df = pd.DataFrame(hotel_data)
places_df = pd.DataFrame(places_data)

# Tool 1: Search Flights
@tool
def search_flights(
    source: str,
    destination: str,
    sort_by: str = "cheapest"
):
    """Search flights between two cities."""

    results = flight_df[
        (flight_df["from"].str.lower() == source.lower()) &
        (flight_df["to"].str.lower() == destination.lower())
    ].copy()

    if results.empty:
        return "No flights found for this route."

    if sort_by.lower() == "cheapest":
        results = results.sort_values(by="price")
    elif sort_by.lower() == "earliest":
        results = results.sort_values(by="departure_time")

    return results.to_dict(orient="records")

# Tool 2: Search Hotels
@tool
def search_hotels(
    city: str,
    min_stars: int = 1,
    sort_by: str = "cheapest"
):
    """Search hotels based on city and minimum star rating."""

    results = hotel_df[
        (hotel_df["city"].str.lower() == city.lower()) &
        (hotel_df["stars"] >= min_stars)
    ].copy()

    if results.empty:
        return "No hotels found matching your requirements."

    if sort_by.lower() == "cheapest":
        results = results.sort_values(by="price_per_night")
    elif sort_by.lower() == "highest_rating":
        results = results.sort_values(
            by="stars", ascending=False
        )

    return results.to_dict(orient="records")

# Tool 3: Search Tourist Places
@tool
def search_places(
    city: str,
    place_type: str = "all",
    min_rating: float = 0
):
    """Search tourist places based on city, type and rating."""

    results = places_df[
        (places_df["city"].str.lower() == city.lower()) &
        (places_df["rating"] >= min_rating)
    ].copy()

    if place_type.lower() != "all":
        results = results[
            results["type"].str.lower() == place_type.lower()
        ]

    if results.empty:
        return "No tourist places found."

    results = results.sort_values(
        by="rating", ascending=False
    )

    return results.to_dict(orient="records")

# Tool 4: Get Weather
@tool
def get_weather():
    """Fetch weather forecast for Goa using Open-Meteo API."""

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": 15.2993,
        "longitude": 74.1240,
        "daily": "temperature_2m_max",
        "timezone": "auto"
    }

    try:
        response = requests.get(
            url, params=params, timeout=20
        )
        response.raise_for_status()

        data = response.json()
        return data["daily"]

    except requests.RequestException as error:
        return {"error": str(error)}

# Tool 5: Estimate Travel Budget
@tool
def estimate_budget(
    flight_price: float,
    hotel_price_per_night: float,
    number_of_nights: int,
    food_cost_per_day: float = 500,
    transport_cost_per_day: float = 300
):
    """Calculate the total estimated travel budget."""

    hotel_cost = hotel_price_per_night * number_of_nights
    food_cost = food_cost_per_day * number_of_nights
    transport_cost = transport_cost_per_day * number_of_nights

    total_budget = (
        flight_price +
        hotel_cost +
        food_cost +
        transport_cost
    )

    return {
        "flight_cost": flight_price,
        "hotel_cost": hotel_cost,
        "food_cost": food_cost,
        "transport_cost": transport_cost,
        "total_budget": total_budget
    }

# Configure Gemini API
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error(
        "Gemini API key not found. "
        "Please add GOOGLE_API_KEY in Streamlit Secrets."
    )
    st.stop()

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=api_key,
    max_retries=0
)

# Register all travel tools
tools = [
    search_flights,
    search_hotels,
    search_places,
    get_weather,
    estimate_budget
]

# Agent instructions
system_prompt = """
You are an intelligent AI travel planning assistant.

Your responsibilities:
1. Understand user travel requirements.
2. Search for suitable flights.
3. Find suitable hotels.
4. Recommend tourist places.
5. Estimate the total travel budget.
6. Use tools whenever relevant.
7. Prepare a clear day-wise itinerary.

Important:
- Use only information returned by the tools.
- Do not invent flight, hotel or tourist place details.
- The weather tool supports Goa only.
- Use weather information only for Goa.
- Clearly mention when weather is unavailable.
- Present prices in Indian Rupees (INR).
- Explain recommendations in simple language.
"""

# Create LangChain agent
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt
)

# Travel Planning Form
st.sidebar.header("🌍 Plan Your Trip")

source = st.sidebar.text_input(
    "Source City",
    placeholder="e.g. Hyderabad"
)

destination = st.sidebar.text_input(
    "Destination City",
    placeholder="e.g. Delhi"
)

number_of_nights = st.sidebar.number_input(
    "Number of Nights",
    min_value=1,
    max_value=30,
    value=3
)

min_stars = st.sidebar.selectbox(
    "Minimum Hotel Rating",
    [1, 2, 3, 4, 5],
    index=2
)

min_rating = st.sidebar.slider(
    "Minimum Tourist Place Rating",
    min_value=0.0,
    max_value=5.0,
    value=4.0,
    step=0.1
)

plan_button = st.sidebar.button(
    "✈️ Generate Travel Plan",
    use_container_width=True
)

# Generate Travel Plan
if plan_button:

    if not source.strip() or not destination.strip():
        st.warning("Please enter both source and destination cities.")

    else:
        user_query = f"""
        Plan a trip from {source} to {destination}
        for {number_of_nights} nights.

        Find:
        1. Cheapest available flight.
        2. Cheapest hotel with at least {min_stars} stars.
        3. Tourist places with a rating of at least {min_rating}.
        4. Weather information if destination is Goa.
        5. Total estimated travel budget.

        Create a simple day-wise itinerary.

        Return the final answer in valid JSON with these keys:
        flight, hotel, tourist_places, weather, budget, itinerary.

        Do not include Markdown formatting or extra text.
        """

        with st.spinner("AI is planning your trip..."):

            try:
                response = agent.invoke({
                    "messages": [
                        {
                            "role": "user",
                            "content": user_query
                        }
                    ]
                })

                raw_output = response["messages"][-1].content

                if isinstance(raw_output, list):
                    raw_output = "\n".join(
                        item["text"]
                        for item in raw_output
                        if item.get("type") == "text"
                    )

                start = raw_output.find("{")
                end = raw_output.rfind("}") + 1

                if start == -1 or end == 0:
                    st.error("AI did not return valid JSON.")

                else:
                    travel_plan = json.loads(
                        raw_output[start:end]
                    )

                    st.session_state["travel_plan"] = travel_plan

            except Exception as error:
                st.error(f"Unable to generate travel plan: {error}")

# Display Travel Plan
if "travel_plan" in st.session_state:

    travel_plan = st.session_state["travel_plan"]

    st.header("✈️ Your Travel Recommendations")

    # Flight Details
    st.subheader("🛫 Flight Details")

    flight = travel_plan["flight"]

    col1, col2, col3 = st.columns(3)

    col1.metric("Airline", flight["airline"])
    col2.metric("Flight Price", f"₹{flight['price']}")
    col3.metric("Flight ID", flight["flight_id"])

    st.write(
        f"**Departure:** {flight['departure_time']}"
    )
    st.write(
        f"**Arrival:** {flight['arrival_time']}"
    )

    st.divider()

    # Hotel Details
    st.subheader("🏨 Hotel Recommendation")

    hotel = travel_plan["hotel"]

    col1, col2, col3 = st.columns(3)

    col1.metric("Hotel", hotel["name"])
    col2.metric("Star Rating", f"{hotel['stars']} ⭐")
    col3.metric(
        "Price per Night",
        f"₹{hotel['price_per_night']}"
    )

    st.write(
        "**Amenities:**",
        ", ".join(hotel["amenities"])
    )

    st.divider()

    # Tourist Places
    st.subheader("📍 Recommended Tourist Places")

    places = travel_plan.get("tourist_places", [])

    if places:
        places_df_display = pd.DataFrame(places)

        st.dataframe(
            places_df_display[
                ["name", "type", "rating"]
            ],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No tourist places found.")

    st.divider()

    # Weather Information
    st.subheader("🌤️ Weather Information")

    st.write(travel_plan.get("weather", "Weather unavailable."))

    st.divider()

    # Budget Details
    st.subheader("💰 Estimated Travel Budget")

    budget = travel_plan["budget"]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Flight Cost",
        f"₹{budget['flight_cost']:,.0f}"
    )

    col2.metric(
        "Hotel Cost",
        f"₹{budget['hotel_cost']:,.0f}"
    )

    col3.metric(
        "Food Cost",
        f"₹{budget['food_cost']:,.0f}"
    )

    col4, col5 = st.columns(2)

    col4.metric(
        "Transport Cost",
        f"₹{budget['transport_cost']:,.0f}"
    )

    col5.metric(
        "Total Budget",
        f"₹{budget['total_budget']:,.0f}"
    )

    st.divider()

    # Day-wise Itinerary
    st.subheader("🗓️ Your Day-Wise Itinerary")

    itinerary = travel_plan.get("itinerary", [])

    for day in itinerary:

        with st.expander(
            f"Day {day['day']}",
            expanded=True
        ):
            st.write("🌅 **Morning:**", day["morning"])
            st.write("☀️ **Afternoon:**", day["afternoon"])
            st.write("🌙 **Evening:**", day["evening"])

    st.divider()

    # Download Travel Plan
    st.subheader("📥 Download Your Travel Plan")

    json_data = json.dumps(
        travel_plan,
        indent=4,
        ensure_ascii=False
    )

    st.download_button(
        label="Download Travel Plan (JSON)",
        data=json_data,
        file_name="travel_plan.json",
        mime="application/json"
    )

    st.success("Your travel plan is ready!")
