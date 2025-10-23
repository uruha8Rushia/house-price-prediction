// static/script.js

// 1) Populate city dropdown on load
document.addEventListener("DOMContentLoaded", () => {
  const citySelect = document.getElementById("city-select");
  CITIES.forEach(city => {
    const option = document.createElement("option");
    option.value = city;
    option.textContent = city;
    citySelect.appendChild(option);
  });

  // Set default city
  citySelect.value = DEFAULT_CITY;
});

// 2) Handle form submission
document.getElementById("predict-form").addEventListener("submit", async (e) => {
  e.preventDefault();

  const formData = new FormData(e.target);
  const data = Object.fromEntries(formData.entries());

  // Convert numeric fields to numbers
  Object.keys(data).forEach(key => {
    if (!isNaN(data[key]) && data[key] !== "") {
      data[key] = Number(data[key]);
    }
  });

  // Remove empty string values so server will use defaults
  Object.keys(data).forEach(k => {
    if (data[k] === "") delete data[k];
  });

  const btn = document.getElementById("predict-btn");
  btn.disabled = true;
  btn.textContent = "Predicting...";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });

    const result = await response.json();

    if (!response.ok) {
      // Show error from server
      alert(result.error || "Server error");
      return;
    }

    // Show result card
    if (result.prediction !== undefined && result.prediction !== null) {
      const priceElement = document.getElementById("price");
      const card = document.getElementById("result-card");

      card.classList.remove("d-none");
      priceElement.textContent = `$ ${Intl.NumberFormat().format(result.prediction)}`;
    }
  } catch (err) {
    alert("An error occurred: " + err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Predict Price";
  }
});
