function random(min, max, decimals = 1) {
    return parseFloat((Math.random() * (max - min) + min).toFixed(decimals));
}

function updateBar(barId, value, min, max, optMin = null, optMax = null) {
    const bar = document.getElementById(barId);
    let percent = ((value - min) / (max - min)) * 100;
    percent = Math.max(0, Math.min(100, percent));
    bar.style.width = percent + "%";

    if (optMin !== null && optMax !== null) {
        bar.style.background =
            value >= optMin && value <= optMax ? "#2ecc71" : "#f1c40f";
    } else {
        bar.style.background = "#3498db";
    }
}

setInterval(() => {

    // Tekanan
    let pressure = random(50, 80);
    document.getElementById("pressureVal").innerText = pressure;
    updateBar("pressureBar", pressure, 50, 80);

    // Suhu Reaktor
    let reactorTemp = random(200, 250);
    document.getElementById("reactorTempVal").innerText = reactorTemp;
    updateBar("reactorTempBar", reactorTemp, 200, 250);

    // Suhu SOEC
    let soecTemp = random(700, 900);
    document.getElementById("soecTempVal").innerText = soecTemp;
    updateBar("soecTempBar", soecTemp, 700, 900);

    // Rasio H2 : CO2
    let ratio = random(2.6, 3.4, 2);
    document.getElementById("ratioVal").innerText = ratio;
    updateBar("ratioBar", ratio, 2.6, 3.4, 2.8, 3.2);

    // Katalis
    let catalyst = random(600, 1800, 0);
    document.getElementById("catalystVal").innerText = catalyst;
    updateBar("catalystBar", catalyst, 600, 1800, 700, 750);

}, 2000);