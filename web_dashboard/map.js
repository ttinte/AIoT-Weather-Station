const MAP_CENTER = { lat: 10.853142956334276, lng: 106.77170424853325 };
const MAP_ZOOM = 14;

function injectMarkerAnimationStyles() {
    if (document.getElementById("marker-animation-styles")) {
        return;
    }

    const style = document.createElement("style");
    style.id = "marker-animation-styles";
    style.textContent = `
        .pulse-marker {
            background: transparent;
            border: none;
        }

        .pulse-marker__wrap {
            position: relative;
            width: 46px;
            height: 46px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .pulse-marker__dot {
            position: relative;
            z-index: 2;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: #3182ce;
            border: 3px solid #ffffff;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.28);
            animation: pulse-marker-bounce 1.8s ease-in-out infinite;
        }

        .pulse-marker__ring {
            position: absolute;
            inset: 8px;
            border-radius: 50%;
            border: 2px solid rgba(49, 130, 206, 0.45);
            animation: pulse-marker-ring 1.8s ease-out infinite;
        }

        @keyframes pulse-marker-ring {
            0% {
                transform: scale(0.45);
                opacity: 0.9;
            }
            70% {
                transform: scale(1.35);
                opacity: 0;
            }
            100% {
                transform: scale(1.35);
                opacity: 0;
            }
        }

        @keyframes pulse-marker-bounce {
            0%, 100% {
                transform: translateY(0);
            }
            50% {
                transform: translateY(-4px);
            }
        }
    `;

    document.head.appendChild(style);
}

function createAnimatedMarkerIcon() {
    return L.divIcon({
        className: "pulse-marker",
        html: `
            <div class="pulse-marker__wrap">
                <div class="pulse-marker__ring"></div>
                <div class="pulse-marker__dot"></div>
            </div>
        `,
        iconSize: [46, 46],
        iconAnchor: [23, 23],
        popupAnchor: [0, -20]
    });
}

function createMap() {
    const mapElement = document.getElementById("map");

    if (!mapElement || !window.L) {
        return null;
    }

    const map = L.map(mapElement).setView([MAP_CENTER.lat, MAP_CENTER.lng], MAP_ZOOM);

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 19
    }).addTo(map);

    return map;
}

function createMarker(map, position) {
    const marker = L.marker([position.lat, position.lng], {
        icon: createAnimatedMarkerIcon()
    }).addTo(map);

    // Dịch lần đầu
    marker.bindPopup(window.getTranslatedText ? window.getTranslatedText("popupContent") : "IoT Station");

    // Lắng nghe khi bấm nút đổi ngôn ngữ
    window.addEventListener('languageChanged', () => {
        marker.setPopupContent(window.getTranslatedText("popupContent"));
    });

    return marker;
}

function initMap() {
    injectMarkerAnimationStyles();

    const map = createMap();

    if (!map) {
        return;
    }

    createMarker(map, MAP_CENTER);
}

document.addEventListener("DOMContentLoaded", function() {
    initMap();
});
