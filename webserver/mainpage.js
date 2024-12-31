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

// ------------ Build containers/layouts ------------
// Define containers
const liked_genres_check_boxes_div = document.getElementById("liked_check_boxes");
const disliked_genres_check_boxes_div = document.getElementById("disliked_check_boxes");
const liked_genres_ranges_div = document.getElementById("genre_proportion_ranges");

// ============ Create check boxes for genres to liked ============
for (let i = 0; i < genres.length; ++i) {
    const check_box = document.createElement("div")
    check_box.innerHTML =`
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
    if (liked_genres.length == 0) { // No genres were chosen
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
                check_box.innerHTML =`
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
    for (let liked_genre of liked_genres) {
        const regulator = document.createElement("div")
        regulator.innerHTML =`
        <label for="liked_genre_${liked_genre}" class="form-label">${genres[liked_genre]}:</label>\
        <input id="liked_genre_${liked_genre}" type="range" class="form-range" min="0" max="100" value="50" step="1" oninput="this.nextElementSibling.value = this.value">\
        <output>50</output>\
        `;
        liked_genres_ranges.push(regulator); // Add it in a list for easier access later
        liked_genres_ranges_div.appendChild(regulator);
    }

    // Remove current layout
    document.getElementById("disliked_genres").style.display = 'none'

    // Show next layout
    document.getElementById("genre_proportions").style.display = 'block'    
});
