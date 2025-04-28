document.addEventListener('DOMContentLoaded', function() {
    // Get form and loading overlay elements
    const uploadForm = document.getElementById('upload-form');
    const loadingOverlay = document.getElementById('loading-overlay');
    
    if (uploadForm) {
        // Add submit event listener to show loading overlay
        uploadForm.addEventListener('submit', function(e) {
            // Validate form first
            const videoInput = document.getElementById('video-file');
            const exerciseType = document.getElementById('exercise-type');
            
            if (!videoInput.files.length) {
                e.preventDefault();
                alert('Please select a video file');
                return;
            }
            
            // Show loading overlay
            loadingOverlay.classList.add('active');
        });
    }
    
    // Add file input change listener to preview file name
    const videoInput = document.getElementById('video-file');
    if (videoInput) {
        videoInput.addEventListener('change', function(e) {
            const fileName = e.target.files[0]?.name || 'No file selected';
            // You can add UI to display the selected filename if desired
            console.log('Selected file:', fileName);
        });
    }
});