// // nav_watcher.js
// console.log("👀 nav_watcher.js started");

// let interval = null;

// async function checkNav() {
//   try {
//     const res = await fetch("/frontend_nav.json?ts=" + Date.now(), { cache: "no-store" });
//     if (res.ok) {
//       const data = await res.json();
//       if (data.navigate_to) {
//         console.log("📦 Worker detected navigation:", data.navigate_to);
//         postMessage({ type: "navigate", url: data.navigate_to });
//       }
//     }
//   } catch (err) {
//     console.warn("⚠️ Worker fetch error:", err.message);
//   }
// }

// onmessage = (e) => {
//   if (e.data === "start") {
//     console.log("🔄 Worker started polling...");
//     // if (interval) clearInterval(interval);
//     // interval = setInterval(checkNav, 1000);
//   } else if (e.data === "stop") {
//     console.log("🛑 Worker stopped.");
//     clearInterval(interval);
//   }
// };
