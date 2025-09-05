// A single Chart.js instance for the modal
let eventChart = null;

const chartColors = {
    red: '#ff5166',
    yellow: '#ffb020',
    green: '#18c964',
    blue: '#5b84ff'
};

function renderChart(data, summary) {
    const ctx = document.getElementById('historyChart').getContext('2d');
    
    const labels = data.map(row => row.x);
    const deltaMins = data.map(row => row.y);

    if (eventChart) {
        eventChart.destroy();
    }

    eventChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Delta (minutes)',
                    data: deltaMins.map((y, i) => ({ x: labels[i], y })),
                    backgroundColor: (context) => {
                        const value = context.raw.y;
                        if (value > summary.upper_bound || value < summary.lower_bound) {
                            return chartColors.red;
                        }
                        return chartColors.green;
                    },
                    pointRadius: 5,
                    pointHoverRadius: 7,
                    borderWidth: 1,
                    borderColor: 'rgba(255, 255, 255, 0.8)',
                },
                {
                    label: 'Mean',
                    data: labels.map(() => ({ x: labels[0], y: summary.mean })),
                    type: 'line',
                    borderColor: chartColors.blue,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    fill: false,
                    tooltip: {
                        callbacks: {
                            label: (context) => `Mean: ${summary.mean.toFixed(2)} min`
                        }
                    }
                },
                {
                    label: 'Expected Range',
                    data: labels.map(() => ({ x: labels[0], y: summary.upper_bound })),
                    type: 'line',
                    borderColor: chartColors.yellow,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    fill: '+1',
                    backgroundColor: 'rgba(255, 176, 32, 0.1)',
                    tooltip: {
                        callbacks: {
                            label: (context) => `Upper Bound: ${summary.upper_bound.toFixed(2)} min`
                        }
                    }
                },
                {
                    label: 'Lower Bound',
                    data: labels.map(() => ({ x: labels[0], y: summary.lower_bound })),
                    type: 'line',
                    borderColor: chartColors.yellow,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    fill: false,
                    tooltip: {
                        callbacks: {
                            label: (context) => `Lower Bound: ${summary.lower_bound.toFixed(2)} min`
                        }
                    }
                },
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'time',
                    time: {
                        tooltipFormat: 'yyyy-MM-dd HH:mm:ss',
                        unit: 'minute'
                    },
                    title: {
                        display: true,
                        text: 'Time'
                    },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Interval (minutes)'
                    },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' }
                }
            },
            plugins: {
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            }
        }
    });
}