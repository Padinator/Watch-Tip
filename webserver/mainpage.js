"use strict";


// Define variables
const genres = [
    "Action",
    "Adventure",
    "Animation",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Family",
    "Fantasy",
    "History",
    "Horror",
    "Music",
    "Mystery",
    "Romance",
    "Science Fiction",
    "TV Movie",
    "Thriller",
    "War",
    "Western",
];
let liked_genres = [];
let disliked_genres = [];
const liked_genres_check_boxes = [];
const disliked_genres_check_boxes = [];
const liked_genres_ranges = [];
const range_init_value = 50;
const bar_chart_font_size = 22;

// ------------ Build containers/layouts ------------
// Define containers
const liked_genres_check_boxes_div = document.getElementById("liked_check_boxes");
const disliked_genres_check_boxes_div = document.getElementById("disliked_check_boxes");
const liked_genres_ranges_div = document.getElementById("genre_proportion_ranges");

// Define variables for the bar chart
const genre_proportion_chart = document.getElementById('genre_proportion_chart');
let bar_chart = null;


function update_bar_chart_by_range(range_obj) {
    const id = range_obj.id;
    const bar_index = id.split("_").at(-1)
    const data = bar_chart.data.datasets[0];

    // Update data of bar chart
    data["data"][bar_index] = range_obj.value;
    bar_chart.update();

    // Update show value
    range_obj.nextElementSibling.value = range_obj.value;
}


// ============ Create check boxes for genres to liked ============
for (let i = 0; i < genres.length; ++i) {
    const check_box = document.createElement("div")
    check_box.innerHTML = `
        <input type="checkbox" class="btn-check" id="btn-check-${i}-outlined" autocomplete="off">
	    <label class="btn btn-outline-secondary" for="btn-check-${i}-outlined">${genres[i]}</label>
    `;
    check_box.classList.add("flex-fill");
    check_box.classList.add("p-2");
    liked_genres_check_boxes.push(check_box); // Add it in  alist for easier access later
    liked_genres_check_boxes_div.appendChild(check_box);
}

// Define function of submit button for liked genres
document.getElementById("submit_liked_genres").addEventListener("click", event => {
    // Find checked genres
    liked_genres = []; // Clear old array of liked genres

    for (let i = 0; i < liked_genres_check_boxes.length; ++i) {
        if (liked_genres_check_boxes[i].children[0].checked) {
            liked_genres.push(i)
        }
    }

    // Update HTML document
    if (liked_genres.length == 0 || 7 < liked_genres.length) { // No genres were chosen
        const error_p = document.getElementById("too_few_liked_genres_chosen");
        error_p.classList.add("text-danger"); // Ephasize error with red text colour

        if (!error_p.innerText.endsWith("!")) {  // Don't append to many "!"
            error_p.innerText += "!";
        }
    } else { // Create next layout with not liked genres
        for (let i = 0; i < liked_genres_check_boxes.length; ++i) {
            if (!liked_genres_check_boxes[i].children[0].checked) {
                // ============ Create check boxes for genres to dislike ============
                const check_box = document.createElement("div")
                check_box.innerHTML = `
                    <input type="checkbox" class="btn-check" id="btn-check-${genres.length + i}-outlined" autocomplete="off">\
                    <label class="btn btn-outline-secondary" for="btn-check-${genres.length + i}-outlined">${genres[i]}</label>\
                `;
                check_box.classList.add("flex-fill");
                check_box.classList.add("p-2");
                disliked_genres_check_boxes.push(check_box); // Add it in a list for easier access later
                disliked_genres_check_boxes_div.appendChild(check_box);
            }
        }

        // Remove current layout
        document.getElementById("liked_genres").style.display = 'none'

        // Show next layout
        document.getElementById("disliked_genres").style.display = 'block'
    }
});

// Define function of submit button for disliked genres
document.getElementById("submit_disliked_genres").addEventListener("click", event => {
    // Save disliked genres in a list
    disliked_genres = []; // Clear old array of disliked genres

    for (let i = 0; i < disliked_genres_check_boxes.length; ++i) {
        if (disliked_genres_check_boxes[i].children[0].checked) {
            disliked_genres.push(i)
        }
    }

    // Show for each liked genre a range/regulator
    for (let index = 0; index < liked_genres.length; ++index) {
        const regulator = document.createElement("div")
        const liked_genre = liked_genres[index]

        // Save liked genre and index for identifying the corresponding genre and bar later
        regulator.innerHTML = `
        <input id="liked_genre_${liked_genre}_${index}" type="range" class="form-range" min="0" max="100" value="50" step="1" oninput="update_bar_chart_by_range(this)">\
        <output style="font-size: ${bar_chart_font_size}px;">${range_init_value}</output>\
        `; // Font size of "output" element is in px, because it's the same one as from the 

        regulator.style = "padding-left: 1em; padding-right: 0.8em;"

        liked_genres_ranges.push(regulator); // Add it in a list for easier access later
        liked_genres_ranges_div.appendChild(regulator);
    }

    // Create chart with all liked genres
    bar_chart = new Chart(genre_proportion_chart, {
        type: 'bar',
        data: {
            labels: liked_genres.map(genre_id => genres[genre_id]),
            datasets: [{
                label: 'Proportion (%) of genres in your perfect movie',
                data: liked_genres.map(_genre => range_init_value),
                borderWidth: 2,
                borderRadius: 10,
                backgroundColor: 'rgba(54, 162, 235, 0.5)',
                borderColor: 'rgba(54, 162, 235, 1)',
                hoverBackgroundColor: 'rgba(75, 192, 192, 0.5)',
                hoverBorderColor: 'rgba(75, 192, 192, 1)'
            }]
        },
        options: {
            scales: {
                x: {
                    ticks: {
                        font: {
                            size: bar_chart_font_size,
                        }
                    }
                },
                y: {
                    beginAtZero: true,
                    ticks: {
                        font: {
                            size: bar_chart_font_size,
                        }
                    }
                }
            }
        }
    });

    // Remove current layout
    document.getElementById("disliked_genres").style.display = 'none'

    // Show next layout
    document.getElementById("genre_proportions").style.display = 'block'
});
