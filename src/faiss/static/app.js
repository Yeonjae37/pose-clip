document.getElementById('searchButton')
.addEventListener('click', function () {
    const searchTerm = document.getElementById('searchInput').value;
    if (searchTerm.trim() !== '') {
        const apiUrl = `/search?search_query=${searchTerm}`;
        fetch(apiUrl)
            .then(response => response.json())
            .then(data => {
                const resultsDiv = document.getElementById('results');
                resultsDiv.innerHTML = '';
                data.forEach(videoUrl => {
                    const video = document.createElement('video');
                    video.src = videoUrl;
                    console.log(videoUrl);
                    video.controls = true;
                    video.style.width = "100%";
                    video.style.height = "auto";
                    resultsDiv.appendChild(video);
                });
            });
    }
});
