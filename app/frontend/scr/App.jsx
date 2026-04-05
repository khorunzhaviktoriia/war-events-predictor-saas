import { useEffect, useState } from "react";
import UkraineMap from "./UkraineMap";

function App() {
  const [region, setRegion] = useState("all");
  const [regions, setRegions] = useState([]);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selectedTime, setSelectedTime] = useState("12:00");

  useEffect(() => {
    const fetchRegions = async () => {
      try {
        const response = await fetch("http://localhost:8000/regions");

        if (!response.ok) {
          throw new Error("Failed to get list of regions");
        }

        const result = await response.json();
        setRegions(result);
      } catch (err) {
        console.error(err);
      }
    };

    fetchRegions();
  }, []);

  const handleGetForecast = async () => {
    try {
      setLoading(true);
      setError("");

      const response = await fetch(
        `http://localhost:8000/forecast?region=${encodeURIComponent(region)}`
      );

      if (!response.ok) {
        throw new Error("Failed to get forecast");
      }

      const result = await response.json();
      setData(result);

      if (result?.regions_forecast) {
        const firstRegion = Object.values(result.regions_forecast)[0];
        const times = firstRegion ? Object.keys(firstRegion) : [];
        if (times.length > 0) {
          setSelectedTime(times[0]);
        }
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleUpdateForecast = async () => {
    try {
      setLoading(true);
      setError("");

      const response = await fetch("http://localhost:8000/forecast/update", {
        method: "POST"
      });

      if (!response.ok) {
        throw new Error("Failed to update forecast");
      }

      await response.json();
      await handleGetForecast();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const availableTimes =
    data?.regions_forecast
      ? Object.keys(Object.values(data.regions_forecast)[0] || {})
      : [];

  return (
    <div className="container">
      <h1>Alarm Forecast</h1>

      <div className="controls">
        <label>Choose the region: </label>
        <select value={region} onChange={(e) => setRegion(e.target.value)}>
          <option value="all">All</option>
          {regions.map((item) => (
            <option key={item.region_id} value={item.region_id}>
            {item.city_name}
          </option>
          ))}
        </select>
      </div>

      <div className="buttons">
        <button onClick={handleGetForecast} disabled={loading}>
          Get forecast
        </button>

        <button onClick={handleUpdateForecast} disabled={loading}>
          Update forecast
        </button>
      </div>

      {loading && <p>Loading...</p>}
      {error && <p className="error">{error}</p>}

      {data && (
        <div className="result">
          <h2>Result</h2>

          <p>
            <strong>Last model train time:</strong>{" "}
            {data.last_model_train_time}
          </p>

          <p>
            <strong>Last prediction time:</strong>{" "}
            {data.last_prediction_time}
          </p>

          {region === "all" && availableTimes.length > 0 && (
            <div className="controls">
              <label>Choose time: </label>
              <select
                value={selectedTime}
                onChange={(e) => setSelectedTime(e.target.value)}
              >
                {availableTimes.map((time) => (
                  <option key={time} value={time}>
                    {time}
                  </option>
                ))}
              </select>
            </div>
          )}

          {region === "all" ? (
            <UkraineMap
              forecastData={data}
              selectedTime={selectedTime}
            />
          ) : (
            <>
              <h3>Forecast</h3>
              <pre>{JSON.stringify(data.regions_forecast, null, 2)}</pre>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
