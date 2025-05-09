// Initialize animations on page load
document.addEventListener('DOMContentLoaded', () => {
    // Animate form elements sequentially
    gsap.from(".form-floating", {
        duration: 0.8,
        y: 50,
        opacity: 0,
        stagger: 0.1,
        ease: "back.out(1.7)"
    });
    
    // Floating car animation
    gsap.to(".car-icon", {
        y: 10,
        duration: 3,
        repeat: -1,
        yoyo: true,
        ease: "sine.inOut"
    });
});

document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const submitBtn = e.target.querySelector('button[type="submit"]');
    const spinner = document.getElementById('spinner');
    const submitText = document.getElementById('submitText');
    const resultContainer = document.getElementById('resultContainer');
    const errorContainer = document.getElementById('errorContainer');
    const formLoader = document.getElementById('formLoader');
    
    // Show loading state with animations
    submitBtn.disabled = true;
    spinner.classList.remove('d-none');
    submitText.textContent = 'ANALYZING...';
    formLoader.classList.remove('d-none');
    errorContainer.classList.add('d-none');
    
    // Animate the button
    gsap.to(submitBtn, {
        scale: 0.95,
        duration: 0.3,
        ease: "power2.out"
    });
    
    try {
        // Get selected model type
        const modelType = document.querySelector('input[name="model_type"]:checked');
        if (!modelType) {
            throw new Error('Please select a model type');
        }

        // Prepare form data
        const formData = new URLSearchParams();
        formData.append('year', document.getElementById('year').value);
        formData.append('mileage', document.getElementById('mileage').value);
        formData.append('transmission', document.getElementById('transmission').value);
        formData.append('fuelType', document.getElementById('fuelType').value);
        formData.append('engine_category', document.getElementById('engine_category').value);
        formData.append('model_type', modelType.value);

        // Show loading animation
        gsap.to(".form-loader", {
            opacity: 1,
            duration: 0.5
        });
        
        // Send to Flask backend
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.status === 'success') {
            // Hide loader
            formLoader.classList.add('d-none');
            
            // Show results with animation
            document.getElementById('predictionResult').textContent = data.prediction;
            
            // Create details HTML
            const detailsHtml = `
                <p><strong>Year:</strong> ${data.details.year}</p>
                <p><strong>Mileage:</strong> ${data.details.mileage.toLocaleString()} miles</p>
                <p><strong>Transmission:</strong> ${data.details.transmission}</p>
                <p><strong>Fuel Type:</strong> ${data.details.fuelType}</p>
                <p><strong>Engine:</strong> ${data.details.engine_category}</p>
            `;
            document.getElementById('predictionDetails').innerHTML = detailsHtml;
            
            // Animate results in
            resultContainer.classList.remove('d-none');
            gsap.from(resultContainer, {
                y: 50,
                opacity: 0,
                duration: 0.8,
                ease: "back.out(1.7)"
            });
            
            // Add confetti effect
            createConfetti();
        } else {
            throw new Error(data.message || 'Unknown error occurred');
        }
    } catch (error) {
        console.error('Prediction error:', error);
        
        // Hide loader
        formLoader.classList.add('d-none');
        
        // Show error with animation
        errorContainer.querySelector('.error-message').textContent = error.message;
        errorContainer.classList.remove('d-none');
        gsap.from(errorContainer, {
            x: -30,
            opacity: 0,
            duration: 0.5,
            ease: "elastic.out(1, 0.5)"
        });
    } finally {
        // Reset button state
        submitBtn.disabled = false;
        spinner.classList.add('d-none');
        submitText.textContent = 'CALCULATE VALUE';
        
        // Animate button back
        gsap.to(submitBtn, {
            scale: 1,
            duration: 0.5,
            ease: "elastic.out(1, 0.5)"
        });
    }
});

// Confetti effect for successful prediction
function createConfetti() {
    const colors = ['#6e00ff', '#00f0ff', '#ffffff', '#ff00aa'];
    const container = document.querySelector('.result-card');
    
    for (let i = 0; i < 50; i++) {
        const confetti = document.createElement('div');
        confetti.className = 'confetti';
        confetti.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)];
        confetti.style.width = `${Math.random() * 8 + 4}px`;
        confetti.style.height = `${Math.random() * 8 + 4}px`;
        confetti.style.left = `${Math.random() * 100}%`;
        container.appendChild(confetti);
        
        gsap.to(confetti, {
            y: -100,
            x: Math.random() * 100 - 50,
            opacity: 0,
            duration: 1 + Math.random() * 2,
            ease: "power1.out",
            onComplete: () => confetti.remove()
        });
    }
}