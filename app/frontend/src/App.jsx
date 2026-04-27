import { useEffect, useMemo, useState } from "react";
import "./index.css";
import UkraineMap from "./UkraineMap.jsx";

const API_BASE_URL = "/api";
const ALARM_THRESHOLD = 0.65;

const isAlarm = (value) => value >= ALARM_THRESHOLD;
const formatDateTime = (value) => {
  if (!value || value === "—") return "—";

  const dt = new Date(value);
  if (Number.isNaN(dt.getTime())) return value;

  return dt.toLocaleString("uk-UA", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
};

function App() {
  const [selectedRegionId, setSelectedRegionId] = useState(null);
  const [region, setRegion] = useState("all");
  const [regions, setRegions] = useState([]);
  const [regionNames, setRegionNames] = useState({});
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [theme, setTheme] = useState("light");

  const getCurrentHour = () => {
    const now = new Date();
  
    const hours = String(now.getHours()).padStart(2, "0");
    return `${hours}:00`;
  };

  const [selectedTime, setSelectedTime] = useState(getCurrentHour());

  const ALWAYS_ALARM_REGIONS = ["1", "12"];
  const REGION_NAME_OVERRIDES = {
    "Zaporozhye": "Zaporizhzhia",
    };
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

  const safeData = data ?? {
    last_model_train_time: "—",
    last_prediction_time: "—",
    forecast_horizon_hours: "—",
    regions_forecast: {},
  };

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
        const original = item.city_name;
        namesMap[String(item.region_id)] =
          REGION_NAME_OVERRIDES[original] || original;
      });
      setRegionNames(namesMap);
    } catch (err) {
      console.error(err);
      setError(err.message || "Failed to load regions");
    }
  };

  fetchRegions();

  handleGetForecast();

}, []);

useEffect(() => {
  const interval = setInterval(() => {
    handleGetForecast();
  }, 5 * 60 * 1000);

  return () => clearInterval(interval);
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

  const availableTimes = useMemo(() => {
    if (!data?.regions_forecast) return [];

    const firstRegion = data.regions_forecast[Object.keys(data.regions_forecast)[0]];
    if (!firstRegion) return [];

    return Object.keys(firstRegion).sort((a, b) => {
      const hourA = parseInt(a.split(":")[0], 10);
      const hourB = parseInt(b.split(":")[0], 10);
      return hourA - hourB;
    });
  }, [data]);

  const rotatedTimes = useMemo(() => {
    if (!availableTimes.length) return [];

    const currentHour = new Date().getHours();

    const index = availableTimes.findIndex(
      (t) => parseInt(t.split(":")[0], 10) === currentHour
    );

    if (index === -1) return availableTimes;

    return [
      ...availableTimes.slice(index),
      ...availableTimes.slice(0, index),
    ];
  }, [availableTimes]);

  const alarmRegionsCount = useMemo(() => {
    if (!data?.regions_forecast || !selectedTime)
      return 0;

    return Object.entries(safeData.regions_forecast).filter(
     ([regionId, forecast]) =>
        ALWAYS_ALARM_REGIONS.includes(regionId) ||
        isAlarm(forecast?.[selectedTime] ?? 0)
    ).length;

  }, [safeData, selectedTime]);

  const totalRegions = 26;

  const peakHourData = useMemo(() => {
    if (!data?.regions_forecast) return { time: "-", count: "-" };

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
    return Math.round(((alarmRegionsCount + 2) / totalRegions) * 100);
  }, [alarmRegionsCount, totalRegions]);

  const hourlyAlarmCounts = useMemo(() => {
    if (!safeData?.regions_forecast) return [];

    return 
availableTimes.map((time) => {
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

    const sortedTimes = Object.entries(forecast || {}).sort((a, b) => {
  const hA = parseInt(a[0].split(":")[0], 10);
  const hB = parseInt(b[0].split(":")[0], 10);
  return hA - hB;
});

const activeHours = sortedTimes
  .filter(
    ([_, value]) =>
      ALWAYS_ALARM_REGIONS.includes(selectedRegionId) || isAlarm(value)
  )
  .map(([time]) => time);

const activeCount = ALWAYS_ALARM_REGIONS.includes(selectedRegionId)
  ? 24
  : activeHours.length;

const currentHour = new Date().getHours();

const nowEntry = sortedTimes.find(([time]) => {
  const hour = parseInt(time.split(":")[0], 10);
  return hour === currentHour;
});

const isNowAlarm =
  nowEntry &&
  (ALWAYS_ALARM_REGIONS.includes(selectedRegionId) ||
    isAlarm(nowEntry[1]));

let firstAlarm = "—";

if (isNowAlarm) {
  firstAlarm = "Now";
} else {
  const nextAlarm = sortedTimes.find(([time, value]) => {
    const hour = parseInt(time.split(":")[0], 10);
    return (
      hour > currentHour &&
      (ALWAYS_ALARM_REGIONS.includes(selectedRegionId) ||
        isAlarm(value))
    );
  });

  if (nextAlarm) {
    firstAlarm = nextAlarm[0];
  } else {
    const fallbackAlarm = sortedTimes.find(
      ([_, value]) =>
        ALWAYS_ALARM_REGIONS.includes(selectedRegionId) ||
        isAlarm(value)
    );

    firstAlarm = fallbackAlarm ? fallbackAlarm[0] : "—";
  }
}

let nextAlarm = "—";

const selectedHour = parseInt(selectedTime.split(":")[0], 10);

const selectedEntry = sortedTimes.find(([time]) => {
  const hour = parseInt(time.split(":")[0], 10);
  return hour === selectedHour;
});

const isSelectedAlarm =
  selectedEntry &&
  (ALWAYS_ALARM_REGIONS.includes(selectedRegionId) ||
    isAlarm(selectedEntry[1]));

if (isSelectedAlarm) {
  nextAlarm = "Now";
} else {
  const nextEntry = sortedTimes.find(([time, value]) => {
    const hour = parseInt(time.split(":")[0], 10);
    return (
      hour > selectedHour &&
      (ALWAYS_ALARM_REGIONS.includes(selectedRegionId) ||
        isAlarm(value))
    );
  });

  if (nextEntry) {
    nextAlarm = nextEntry[0];
  } else {
    const fallbackEntry = sortedTimes.find(
      ([_, value]) =>
        ALWAYS_ALARM_REGIONS.includes(selectedRegionId) ||
        isAlarm(value)
    );

    nextAlarm = fallbackEntry ? fallbackEntry[0] : "—";
  }
}

return {
  regionId: selectedRegionId,
  name: regionNames[selectedRegionId] || `Region ${selectedRegionId}`,
  activeHours,
  activeCount,
  coverage: Math.round((activeCount / 24) * 100),
  firstAlarm,
  nextAlarm,
};
  }, [selectedRegionId, safeData, regionNames]);

  const mostAffectedRegions = useMemo(() => {
  if (!safeData?.regions_forecast || !selectedTime) return [];

  return Object.entries(safeData.regions_forecast)
    .map(([regionId, forecast]) => ({
      regionId,
      name: regionNames[regionId] || `Region ${regionId}`,
      probability:
        ALWAYS_ALARM_REGIONS.includes(regionId)
          ? 1
          : forecast?.[selectedTime] ?? 0,
    }))
    .sort((a, b) => b.probability - a.probability)
    .slice(0, 3);
}, [safeData, regionNames, selectedTime]);

  return (
    <div className="container">
      <div className="page-header">
        <div>
          <h1>Alarm Forecast</h1>
          <p>24-hour forecast of air alarms by region. Forecasts are generated every hour. To ensure you are viewing the latest data, please refresh the page approximately 15 minutes after the beginning of a new hour. </p>
        </div>

        <div className="top-buttons">
          
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
                <div className="card-title">Forecast map at {selectedTime}</div>
                <p className="muted">Predicted probability of air alarm by region</p>
              </div>

              <div className="muted">
                <strong>{alarmRegionsCount+2}</strong> alarms
              </div>
            </div>

            <div className="map-shell">
              <UkraineMap
                forecastData={data ? data.regions_forecast : {}}
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
              <span>A.R. Crimea, Luhansk are considered to be regions with constant alarm</span>
            </div>
          </div>

          <div className="card">
            <div className="card-title-row">
              <span className="card-title">Hour Selector</span>
              <span className="muted">{selectedTime}</span>
            </div>

            {region === "all" && availableTimes.length > 0 && (
              <div className="time-scroll">
                {rotatedTimes.map((time) => {
                  const count =
 		   Object.entries(data?.regions_forecast || {}).filter(
		    ([regionId, forecast]) =>
		      ALWAYS_ALARM_REGIONS.includes(regionId) ||
		      isAlarm(forecast?.[time] ?? 0)
		  ).length + 2;

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
          <div className="stat-label">First Predicted  Alarm</div>
          <div className="stat-value" style={{ fontSize: "22px" }}>
            {selectedRegionStats.firstAlarm}
          </div>
        </div>

        <div className="stat-box">
 	  <div className="stat-label">Next Alarm</div>
 	  <div className="stat-value">
	    {selectedRegionStats.nextAlarm === "Now" ? (
  <span style={{ color: "#ef4444", fontWeight: 700 }}>
    Now
  </span>
) : (
  selectedRegionStats.nextAlarm
)}
	  </div>
	</div>
      </div>

      <div style={{ marginTop: "18px" }}>
        <button 
	  className="reset-button"
	  onClick={() => setSelectedRegionId(null)}>
          Reset region selection
        </button>
      </div>
    </div>
  ) : (
    <>
      {/* STATISTICS */}
      <div className="card">
        <div className="card-title-row">
          <span className="card-title">Statistics</span>
          <span className="muted">{selectedTime}</span>
        </div>

        <div className="stats-grid">
          <div className="stat-box red">
            <div className="stat-label">Alarm Regions</div>
            <div className="stat-value">{data ? (alarmRegionsCount + 2) : "-"}</div>
            <div className="muted">{data ? `of ${totalRegions} total` : "-"}</div>
          </div>

          <div className="stat-box green">
            <div className="stat-label">Clear Regions</div>
            <div className="stat-value">
              {data ? Math.max(0, totalRegions - alarmRegionsCount - 2) : "-"}
            </div>
            <div className="muted">of {totalRegions} total</div>
          </div>

          <div className="stat-box">
            <div className="stat-label">Peak Hour</div>
            <div className="stat-value" style={{ fontSize: "22px" }}>
              {data ? peakHourData.time : "-"}
            </div>
            <div className="muted">{data ? `${peakHourData.count} regions` : "-"}</div>
          </div>

          <div className="stat-box">
            <div className="stat-label">Coverage</div>
            <div className="stat-value">{data ? `${coveragePercent}%` : "-"}</div>
            <div className="muted">of all regions</div>
          </div>
        </div>

	<p className="muted" style={{ marginTop: "6px", fontSize: "12px" }}>
            Regions with probability ≥ 0.65 (red and dark-red) are considered alarm regions.
        </p>

        <div style={{ marginTop: "18px" }}>
          <div className="card-title">
            Regions with the highest probability of alarm at {selectedTime}
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "10px", marginTop: "10px" }}>
            {mostAffectedRegions.map((item, index) => (
              <div key={item.regionId}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "4px", fontSize: "13px" }}>
                  <span>{index + 1}. {item.name}</span>
                  <span>
                    {Math.round(item.probability * 100)}%
                  </span>
                </div>

                <div style={{ height: "6px", background: "#e5e7eb", borderRadius: "999px", overflow: "hidden" }}>
                  <div style={{ width: `${item.probability * 100}%`, height: "100%", background: "#ef4444" }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* MODEL */}
      <div className="card">
        <div className="card-title">Model</div>

        <div className="info-list">
          <div className="info-row">
            <span className="info-label">Training:</span>
            <span className="info-value">
              {formatDateTime(safeData.last_model_train_time)}
            </span>
          </div>

          <div className="info-row">
            <span className="info-label">Forecast:</span>
            <span className="info-value">
              {formatDateTime(safeData.last_prediction_time)}
            </span>
          </div>

          <div className="info-row">
            <span className="info-label">Horizon:</span>
            <span className="info-value">
              {safeData.forecast_horizon_hours || 24} hours
            </span>
          </div>
        </div>
      </div>

      {/* DEVELOPERS */}
      <div className="card">
        <div className="card-title">Developers</div>
        <p className="developers-text">
          Developed in March–April 2026 for educational purposes by NaUKMA students
          Viktoriia Khorunzha, Polina Tkachenko, Mariia Vakorina, and Slava Veles.
        </p>
      </div>
    </>
  )}
</div>
      </div>
    </div>
  );
}

export default App;
