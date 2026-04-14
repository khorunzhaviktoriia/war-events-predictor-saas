import { useEffect, useMemo, useState } from "react";
import "./index.css";
import UkraineMap from "./UkraineMap";

const API_BASE_URL = "http://127.0.0.1:8000";

function App() {
  const [selectedRegionId, setSelectedRegionId] = useState(null);
  const [region, setRegion] = useState("all");
  const [regions, setRegions] = useState([]);
  const [regionNames, setRegionNames] = useState({});
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [theme, setTheme] = useState("light");
  const [selectedTime, setSelectedTime] = useState("12:00");

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

      const response = await fetch(
        `${API_BASE_URL}/forecast?region=${encodeURIComponent(region)}`
      );

      if (!response.ok) {
        throw new Error("Failed to get forecast");
      }

      const result = await response.json();
      setData(result);

      if (result?.regions_forecast) {
        const firstRegion = Object.values(result.regions_forecast)[0];
        const times = firstRegion ? Object.keys(firstRegion) : [];

        if (times.length > 0 && !times.includes(selectedTime)) {
          setSelectedTime(times[0]);
        }
      }
    } catch (err) {
      console.error(err);
      setError(err.message || "Failed to get forecast");
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
    const firstRegion = Object.values(data.regions_forecast)[0];
    return firstRegion ? Object.keys(firstRegion) : [];
  }, [data]);

  const alarmRegionsCount = useMemo(() => {
    if (!data?.regions_forecast || !selectedTime) return 0;

    return Object.values(data.regions_forecast).filter(
      (regionForecast) => regionForecast?.[selectedTime] === true
    ).length;
  }, [data, selectedTime]);

  const totalRegions = useMemo(() => {
    if (data?.regions_forecast) {
      return Object.keys(data.regions_forecast).length;
    }
    return regions.length;
  }, [data, regions]);

  const peakHourData = useMemo(() => {
    if (!data?.regions_forecast) return { time: "-", count: 0 };

    let bestTime = "-";
    let bestCount = -1;

    availableTimes.forEach((time) => {
      const count = Object.values(data.regions_forecast).filter(
        (forecast) => forecast?.[time] === true
      ).length;

      if (count > bestCount) {
        bestCount = count;
        bestTime = time;
      }
    });

    return { time: bestTime, count: bestCount };
  }, [data, availableTimes]);

  const coveragePercent = useMemo(() => {
    if (!totalRegions) return 0;
    return Math.round((alarmRegionsCount / totalRegions) * 100);
  }, [alarmRegionsCount, totalRegions]);

  const hourlyAlarmCounts = useMemo(() => {
    if (!data?.regions_forecast) return [];

    return availableTimes.map((time) => {
      const count = Object.values(data.regions_forecast).filter(
        (forecast) => forecast?.[time] === true
      ).length;

      return { time, count };
    });
  }, [data, availableTimes]);

  const mostAffectedRegions = useMemo(() => {
    if (!data?.regions_forecast) return [];

    return Object.entries(data.regions_forecast)
      .map(([regionId, forecast]) => {
        const alarmHours = Object.values(forecast).filter(Boolean).length;
        return {
          regionId,
          name: regionNames[regionId] || `Region ${regionId}`,
          alarmHours,
        };
      })
      .sort((a, b) => b.alarmHours - a.alarmHours)
      .slice(0, 3);
  }, [data, regionNames]);

  const selectedRegionStats = useMemo(() => {
    if (!selectedRegionId || !data?.regions_forecast?.[selectedRegionId]) return null;

    const forecast = data.regions_forecast[selectedRegionId];
    const activeHours = Object.entries(forecast)
      .filter(([, value]) => value === true)
      .map(([time]) => time);

    return {
      regionId: selectedRegionId,
      name: regionNames[selectedRegionId] || `Region ${selectedRegionId}`,
      activeHours,
      activeCount: activeHours.length,
      coverage: Math.round((activeHours.length / 24) * 100),
      firstAlarm: activeHours[0] || "—",
    };
  }, [selectedRegionId, data, regionNames]);

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

      {data && (
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
                  forecastData={data.regions_forecast}
                  selectedTime={selectedTime}
                  theme={theme}
                  regionNames={regionNames}
                  selectedRegionId={selectedRegionId}
                  onRegionClick={setSelectedRegionId}
                />
              </div>

              <div className="legend">
                <div className="legend-item">
                  <div className="legend-box legend-red"></div>
                  <span>Alarm Expected</span>
                </div>

                <div className="legend-item">
                  <div className="legend-box legend-gray"></div>
                  <span>No Alarm</span>
                </div>
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
                    const count = Object.values(data.regions_forecast).filter(
                      (forecast) => forecast?.[time] === true
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
                      const isActive = data?.regions_forecast?.[selectedRegionId]?.[time] === true;
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
                  <div className="stat-label">Most affected regions</div>
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
                          <span>{item.alarmHours}h</span>
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
                              width: `${(item.alarmHours / 24) * 100}%`,
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
      )}
    </div>
  );
}

export default App;