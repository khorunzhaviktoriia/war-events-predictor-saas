import { useEffect, useMemo, useState } from "react";
import "./index.css";
import UkraineMap from "./UkraineMap.jsx";

const API_BASE_URL = "";
const ALARM_THRESHOLD = 0.5;

const isAlarm = (value) => value >= ALARM_THRESHOLD;

function App() {
  const [selectedRegionId, setSelectedRegionId] = useState(null);
  const [region, setRegion] = useState("all");
  const [regions, setRegions] = useState([]);
  const [regionNames, setRegionNames] = useState({});
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [theme, setTheme] = useState("light");

  const getNextHour = () => {
    const now = new Date();
    now.setHours(now.getHours() + 1);

    const hours = String(now.getHours()).padStart(2, "0");
    return `${hours}:00`;
  };

  const [selectedTime, setSelectedTime] = useState(getNextHour());

  const ALWAYS_ALARM_REGIONS = ["1", "12"];

  const fallbackForecast = {
    last_model_train_time: "—",
    last_prediction_time: "—",
    model_name: "demo_mode",
    forecast_horizon_hours: 24,
    regions_forecast: {
      "1": { "12:00": 0 },
      "2": { "12:00": 0 },
      "3": { "12:00": 0 },
      "4": { "12:00": 0 },
      "5": { "12:00": 0 },
      "6": { "12:00": 0 },
      "7": { "12:00": 0 },
      "8": { "12:00": 0 },
      "9": { "12:00": 0 },
      "10": { "12:00": 0 },
      "11": { "12:00": 0 },
      "12": { "12:00": 0 },
      "13": { "12:00": 0 },
      "14": { "12:00": 0 },
      "15": { "12:00": 0 },
      "16": { "12:00": 0 },
      "17": { "12:00": 0 },
      "18": { "12:00": 0 },
      "19": { "12:00": 0 },
      "20": { "12:00": 0 },
      "21": { "12:00": 0 },
      "22": { "12:00": 0 },
      "23": { "12:00": 0 },
      "24": { "12:00": 0 },
      "25": { "12:00": 0 },
      "26": { "12:00": 0 },
    },
  };

  const safeData = data || fallbackForecast;

  useEffect(() => {
    document.body.className = theme;
  }, [theme]);

  useEffect(() => {
    const fetchRegions = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/regions`);

        if (!response.ok) {
          throw new Error("Failed to get list of regions");
        }

        const result = await response.json();
        setRegions(result);

        const namesMap = {};
        result.forEach((item) => {
          namesMap[String(item.region_id)] = item.city_name;
        });
        setRegionNames(namesMap);
      } catch (err) {
        console.error(err);
        setError(err.message || "Failed to load regions");
      }
    };

    fetchRegions();
  }, []);

  const handleGetForecast = async () => {
    try {
      setLoading(true);
      setError("");

      const response = await fetch(`${API_BASE_URL}/forecast`);

      if (!response.ok) throw new Error("Failed to fetch");

      const result = await response.json();
      setData(result);

    } catch (err) {
      console.error(err);
      setError("Backend unavailable");
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  const handleUpdateForecast = async () => {
    try {
      setLoading(true);
      setError("");

      const response = await fetch(`${API_BASE_URL}/forecast/update`, {
        method: "POST",
      });

      if (!response.ok) {
        throw new Error("Failed to update forecast");
      }

      await response.json();
      await handleGetForecast();
    } catch (err) {
      console.error(err);
      setError(err.message || "Failed to update forecast");
    } finally {
      setLoading(false);
    }
  };

  const availableTimes = useMemo(() => {
    if (!data?.regions_forecast) return [];

    const firstRegion = data?.regions_forecast
    ? data.regions_forecast[Object.keys(data.regions_forecast)[0]]
    : null;
    if (!firstRegion) return [];

    return Object.keys(firstRegion).sort((a, b) => {
      const hourA = parseInt(a.split(":")[0], 10);
      const hourB = parseInt(b.split(":")[0], 10);
      return hourA - hourB;
    });
  }, [data]);

  const alarmRegionsCount = useMemo(() => {
    if (!safeData?.regions_forecast || !selectedTime)
      return ALWAYS_ALARM_REGIONS.length;

    const dynamicCount = Object.entries(safeData.regions_forecast).filter(
      ([regionId, forecast]) =>
        !ALWAYS_ALARM_REGIONS.includes(regionId) &&
        isAlarm((forecast?.[selectedTime] ?? 0))
    ).length;

    return dynamicCount + ALWAYS_ALARM_REGIONS.length;
  }, [safeData, selectedTime]);

  const totalRegions = useMemo(() => {
    if (data?.regions_forecast) {
      return Object.keys(data.regions_forecast).length;
    }
    return regions.length;
  }, [data, regions]);

  const peakHourData = useMemo(() => {
    if (!safeData?.regions_forecast) return { time: "-", count: 0 };

    let bestTime = "-";
    let bestCount = -1;

    availableTimes.forEach((time) => {
      const count = Object.entries(safeData.regions_forecast).filter(
        ([regionId, forecast]) =>
          ALWAYS_ALARM_REGIONS.includes(regionId) ||
          isAlarm(forecast?.[time])
      ).length;

      if (count > bestCount) {
        bestCount = count;
        bestTime = time;
      }
    });

    return { time: bestTime, count: bestCount };
  }, [safeData, availableTimes]);

  const coveragePercent = useMemo(() => {
    if (!totalRegions) return 0;
    return Math.round((alarmRegionsCount / totalRegions) * 100);
  }, [alarmRegionsCount, totalRegions]);

  const hourlyAlarmCounts = useMemo(() => {
    if (!safeData?.regions_forecast) return [];

    return availableTimes.map((time) => {
      const count = Object.values(safeData.regions_forecast).filter(
        (forecast) => isAlarm(forecast?.[time])
      ).length;

      return { time, count };
    });
  }, [safeData, availableTimes]);

  const selectedRegionStats = useMemo(() => {
    if (!selectedRegionId || !safeData?.regions_forecast?.[selectedRegionId])
      return null;

    const forecast = safeData.regions_forecast[selectedRegionId];

    const activeHours = Object.entries(forecast)
      .filter(
        ([_, value]) =>
          ALWAYS_ALARM_REGIONS.includes(selectedRegionId) || isAlarm(value)
      )
      .map(([time]) => time);

    const activeCount = ALWAYS_ALARM_REGIONS.includes(selectedRegionId)
      ? 24
      : activeHours.length;

    return {
      regionId: selectedRegionId,
      name: regionNames[selectedRegionId] || `Region ${selectedRegionId}`,
      activeHours,
      activeCount,
      coverage: Math.round((activeCount / 24) * 100),
      firstAlarm: activeHours[0] || "—",
    };
  }, [selectedRegionId, safeData, regionNames]);

  const mostAffectedRegions = useMemo(() => {
    if (!safeData?.regions_forecast) return [];

    return Object.entries(safeData.regions_forecast)
      .map(([regionId, forecast]) => {
        const values = Object.values(forecast).map(v => v || 0);

        const avg =
          values.reduce((a, b) => a + b, 0) / values.length;

        const max = Math.max(...values);

        const variance =
          values.reduce((sum, v) => sum + Math.pow(v - avg, 2), 0) /
          values.length;

        const volatility = Math.sqrt(variance); // σ

        const riskScore = 0.5 * max + 0.3 * avg + 0.2 * volatility;

        const adjustedScore = ALWAYS_ALARM_REGIONS.includes(regionId)
          ? 1
          : riskScore;

        return {
          regionId,
          name: regionNames[regionId] || `Region ${regionId}`,
          probability: adjustedScore,
          max,
          avg,
          volatility,
        };
      })
      .sort((a, b) => b.probability - a.probability)
      .slice(0, 3);
  }, [safeData, regionNames]);

  return (
    <div className="container">
      <div className="page-header">
        <div>
          <h1>Alarm Forecast</h1>
          <p>24-hour forecast of air alarms by region</p>
        </div>

        <div className="top-buttons">
          <button onClick={handleGetForecast} disabled={loading}>
            Get forecast
          </button>

          <button onClick={handleUpdateForecast} disabled={loading}>
            Update forecast
          </button>

          <button onClick={() => setTheme(theme === "light" ? "dark" : "light")}>
            {theme === "light" ? "🌙 Dark" : "☀️ Light"}
          </button>
        </div>
      </div>

      {loading && <p className="muted">Loading...</p>}
      {error && <p className="error">{error}</p>}

      <div className="layout">
        <div className="left-column">
          <div className="card">
            <div className="card-title-row">
              <div>
                <div className="card-title">Interactive Map</div>
                <p className="muted">Selected hour: {selectedTime}</p>
              </div>

              <div className="muted">
                <strong>{alarmRegionsCount}</strong> alarms
              </div>
            </div>

            <div className="map-shell">
              <UkraineMap
                forecastData={safeData.regions_forecast}
                selectedTime={selectedTime}
                theme={theme}
                regionNames={regionNames}
                selectedRegionId={selectedRegionId}
                onRegionClick={setSelectedRegionId}
              />
            </div>

           <div className="legend">
            <div className="legend-item">
              <div className="legend-box" style={{ background: "#dbeafe" }}></div>
              <span>0.00 – 0.20 (Very low)</span>
            </div>

            <div className="legend-item">
              <div className="legend-box" style={{ background: "#86efac" }}></div>
              <span>0.20 – 0.35 (Low)</span>
            </div>

            <div className="legend-item">
              <div className="legend-box" style={{ background: "#fde68a" }}></div>
              <span>0.35 – 0.50 (Moderate)</span>
            </div>

            <div className="legend-item">
              <div className="legend-box" style={{ background: "#fdba74" }}></div>
              <span>0.50 – 0.65 (Elevated)</span>
            </div>

            <div className="legend-item">
              <div className="legend-box" style={{ background: "#fb7185" }}></div>
              <span>0.65 – 0.80 (High)</span>
            </div>

            <div className="legend-item">
              <div className="legend-box" style={{ background: "#b91c1c" }}></div>
              <span>0.80 – 1.00 (Critical)</span>
            </div>

            <div className="legend-item">
              <div className="legend-box" style={{ background: "#d9d9d9" }}></div>
              <span>No data</span>
            </div>
          </div>

          <div className="legend-item">
              <div className="legend-box" style={{ background: "#d62828" }}></div>
              <span>A.R. Crimea, Luhansk</span>
            </div>
          </div>

          <div className="card">
            <div className="card-title-row">
              <span className="card-title">Hour Selector</span>
              <span className="muted">{selectedTime}</span>
            </div>

            {region === "all" && availableTimes.length > 0 && (
              <div className="time-scroll">
                {availableTimes.map((time) => {
                  const count = Object.values(data?.regions_forecast || {}).filter(
                    (forecast) => forecast?.[time] >= 0.5
                  ).length;

                  const isSelected = selectedTime === time;

                  return (
                    <button
                      key={time}
                      onClick={() => setSelectedTime(time)}
                      className={`time-button ${isSelected ? "active" : ""}`}
                      title={`${time} — ${count} alarms`}
                    >
                      <div className="time-hour">{time.slice(0, 2)}</div>
                      <div className="time-count">{count}</div>
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        </div>

        <div className="right-column">
          {selectedRegionStats ? (
            <div className="card">
              <div className="card-title-row">
                <span className="card-title">{selectedRegionStats.name}</span>
                <span className="muted">{selectedTime}</span>
              </div>

              <div className="stats-grid">
                <div className="stat-box red">
                  <div className="stat-label">Alarm Hours</div>
                  <div className="stat-value">{selectedRegionStats.activeCount}</div>
                  <div className="muted">of 24 total</div>
                </div>

                <div className="stat-box green">
                  <div className="stat-label">Day Cover</div>
                  <div className="stat-value">{selectedRegionStats.coverage}%</div>
                  <div className="muted">of the day</div>
                </div>

                <div className="stat-box">
                  <div className="stat-label">First Alarm</div>
                  <div className="stat-value" style={{ fontSize: "22px" }}>
                    {selectedRegionStats.firstAlarm}
                  </div>
                </div>
              </div>

              <div style={{ marginTop: "16px" }}>
                <div className="stat-label">Alarm windows</div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: "6px", marginTop: "8px" }}>
                  {selectedRegionStats.activeHours.length > 0 ? (
                    selectedRegionStats.activeHours.map((time) => (
                      <span
                        key={time}
                        style={{
                          padding: "6px 8px",
                          borderRadius: "8px",
                          background: "#fee2e2",
                          color: "#dc2626",
                          fontSize: "12px",
                          fontWeight: 600,
                        }}
                      >
                        {time}
                      </span>
                    ))
                  ) : (
                    <span className="muted">No alarms predicted</span>
                  )}
                </div>
              </div>

              <div style={{ marginTop: "18px" }}>
                <div className="stat-label">24h Timeline</div>
                <div style={{ display: "flex", gap: "4px", marginTop: "10px" }}>
                  {availableTimes.map((time) => {
                    const isActive = data?.regions_forecast?.[selectedRegionId]?.[time] >= 0.5;
                    const isCurrent = time === selectedTime;

                    return (
                      <div
                        key={time}
                        title={time}
                        style={{
                          flex: 1,
                          height: "30px",
                          borderRadius: "4px",
                          background: isActive ? "#ef4444" : "#e5e7eb",
                          outline: isCurrent ? "2px solid #111827" : "none",
                        }}
                      />
                    );
                  })}
                </div>
              </div>

              <div style={{ marginTop: "18px" }}>
                <button onClick={() => setSelectedRegionId(null)}>
                  Reset region selection
                </button>
              </div>
            </div>
          ) : (
            <div className="card">
              <div className="card-title-row">
                <span className="card-title">Statistics</span>
                <span className="muted">{selectedTime}</span>
              </div>

              <div className="stats-grid">
                <div className="stat-box red">
                  <div className="stat-label">Alarm Regions</div>
                  <div className="stat-value">{alarmRegionsCount}</div>
                  <div className="muted">of {totalRegions} total</div>
                </div>

                <div className="stat-box green">
                  <div className="stat-label">Clear Regions</div>
                  <div className="stat-value">{Math.max(0, totalRegions - alarmRegionsCount)}</div>
                  <div className="muted">of {totalRegions} total</div>
                </div>

                <div className="stat-box">
                  <div className="stat-label">Peak Hour</div>
                  <div className="stat-value" style={{ fontSize: "22px" }}>
                    {peakHourData.time}
                  </div>
                  <div className="muted">{peakHourData.count} regions</div>
                </div>

                <div className="stat-box">
                  <div className="stat-label">Coverage</div>
                  <div className="stat-value">{coveragePercent}%</div>
                  <div className="muted">of all regions</div>
                </div>
              </div>

              <div style={{ marginTop: "18px" }}>
                <div className="stat-label">Alarm coverage</div>
                <div
                  style={{
                    height: "8px",
                    background: "#e5e7eb",
                    borderRadius: "999px",
                    overflow: "hidden",
                    marginTop: "8px",
                  }}
                >
                  <div
                    style={{
                      width: `${coveragePercent}%`,
                      height: "100%",
                      background: "#ef4444",
                    }}
                  />
                </div>
              </div>

              <div style={{ marginTop: "18px" }}>
                <div className="stat-label">24h alarm intensity</div>
                <div style={{ display: "flex", gap: "2px", marginTop: "10px" }}>
                  {hourlyAlarmCounts.map(({ time, count }) => {
                    const opacity = count === 0 ? 0.2 : Math.max(0.25, count / totalRegions);
                    const isCurrent = time === selectedTime;

                    return (
                      <div
                        key={time}
                        title={`${time} — ${count} regions`}
                        style={{
                          flex: 1,
                          height: "26px",
                          borderRadius: "4px",
                          background: `rgba(239,68,68,${opacity})`,
                          outline: isCurrent ? "2px solid #111827" : "none",
                        }}
                      />
                    );
                  })}
                </div>
              </div>

              <div style={{ marginTop: "18px" }}>
                <div className="stat-label">Regions with the highest probability of alarm</div>
                <div style={{ display: "flex", flexDirection: "column", gap: "10px", marginTop: "10px" }}>
                  {mostAffectedRegions.map((item, index) => (
                    <div key={item.regionId}>
                      <div
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                          marginBottom: "4px",
                          fontSize: "13px",
                        }}
                      >
                        <span>{index + 1}. {item.name}</span>
                        <span>
                          {Math.round(item.probability * 100)}% 
                          (σ {item.volatility.toFixed(2)})
                        </span>
                      </div>
                      <div
                        style={{
                          height: "6px",
                          background: "#e5e7eb",
                          borderRadius: "999px",
                          overflow: "hidden",
                        }}
                      >
                        <div
                          style={{
                            width: `${item.probability * 100}%`,
                            height: "100%",
                            background: "#ef4444",
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;