// Chat Functionality
async function sendMessage() {
    const input = document.getElementById('user-input');
    const message = input.value.trim();
    if (!message) return;

    const history = document.getElementById('chat-history');
    
    // Add user message
    history.innerHTML += `
        <div class="message user-message">
            <div class="message-content">${message}</div>
        </div>
    `;
    
    input.value = '';
    history.scrollTop = history.scrollHeight;

    // Add loading indicator
    const loadingId = 'loading-' + Date.now();
    history.innerHTML += `
        <div class="message bot-message" id="${loadingId}">
            <div class="message-content">Thinking... 🤔</div>
        </div>
    `;
    history.scrollTop = history.scrollHeight;

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message })
        });
        
        const data = await response.json();
        const loadingEl = document.getElementById(loadingId);
        loadingEl.remove();

        if (data.error) {
            history.innerHTML += `
                <div class="message bot-message">
                    <div class="message-content">❌ Error: ${data.error}</div>
                </div>
            `;
        } else {
            let content = `<div class="message-content">${data.response}</div>`;
            
            if (data.sql_query) {
                content += `<div class="sql-block"><strong>SQL:</strong> ${data.sql_query}</div>`;
            }
            
            if (data.sql_result) {
                content += `<div class="raw-result"><strong>Raw Result:</strong>\n${data.sql_result}</div>`;
            }

            history.innerHTML += `
                <div class="message bot-message">
                    ${content}
                </div>
            `;
        }
    } catch (error) {
        console.error('Error:', error);
    }
    history.scrollTop = history.scrollHeight;
}

function handleKeyPress(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

// Analytics Functionality
async function loadAnalytics() {
    console.log('Loading analytics data...');
    try {
        const response = await fetch('/api/analytics-data');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        console.log('Analytics data received:', data);

        // Check if data is valid
        if (!data || Object.keys(data).length === 0) {
            console.error('No data received from API');
            return;
        }

        // 1. Wins by Team (Bar)
        if (data.wins_by_team && data.wins_by_team.length > 0) {
            Plotly.newPlot('chart-wins', [{
                x: data.wins_by_team.map(d => d.Wins),
                y: data.wins_by_team.map(d => d.Team),
                type: 'bar',
                orientation: 'h',
                marker: { color: '#1565C0' }
            }], { margin: { t: 0, l: 150 } });
        }

        // 2. Toss Decision (Pie)
        if (data.toss_decision && data.toss_decision.length > 0) {
            Plotly.newPlot('chart-toss', [{
                values: data.toss_decision.map(d => d.Count),
                labels: data.toss_decision.map(d => d.Decision),
                type: 'pie',
                marker: { colors: ['#1976D2', '#64B5F6'] }
            }], { margin: { t: 0 } });
        }

        // 3. Matches by Type (Bar)
        if (data.matches_by_type && data.matches_by_type.length > 0) {
            Plotly.newPlot('chart-type', [{
                x: data.matches_by_type.map(d => d.Type),
                y: data.matches_by_type.map(d => d.Matches),
                type: 'bar',
                marker: { color: '#0D47A1' }
            }], { margin: { t: 0 } });
        }

        // 4. Top Scorers (Bar)
        if (data.top_scorers && data.top_scorers.length > 0) {
            Plotly.newPlot('chart-scorers', [{
                x: data.top_scorers.map(d => d.Runs),
                y: data.top_scorers.map(d => d.Player),
                type: 'bar',
                orientation: 'h',
                marker: { color: '#1E88E5' }
            }], { margin: { t: 0, l: 150 } });
        }

        // 5. Matches per Season (Line)
        if (data.matches_per_season && data.matches_per_season.length > 0) {
            Plotly.newPlot('chart-season', [{
                x: data.matches_per_season.map(d => d.Season),
                y: data.matches_per_season.map(d => d.Matches),
                type: 'scatter',
                mode: 'lines+markers',
                line: { color: '#1565C0', width: 3 }
            }], { margin: { t: 20 } });
        }

        // 6. Win Method (Pie)
        if (data.win_method && data.win_method.length > 0) {
            Plotly.newPlot('chart-method', [{
                values: data.win_method.map(d => d.Count),
                labels: data.win_method.map(d => d.Method),
                type: 'pie',
                marker: { colors: ['#1565C0', '#42A5F5', '#90CAF9'] }
            }], { margin: { t: 0 } });
        }

        // 7. Top Wicket Takers (Bar)
        if (data.top_wicket_takers && data.top_wicket_takers.length > 0) {
            Plotly.newPlot('chart-wickets', [{
                x: data.top_wicket_takers.map(d => d.Wickets),
                y: data.top_wicket_takers.map(d => d.Player),
                type: 'bar',
                orientation: 'h',
                marker: { color: '#0D47A1' }
            }], { margin: { t: 0, l: 150 } });
        }

        // 8. Matches by City (Pie)
        if (data.matches_by_city && data.matches_by_city.length > 0) {
            Plotly.newPlot('chart-city', [{
                values: data.matches_by_city.map(d => d.Matches),
                labels: data.matches_by_city.map(d => d.City),
                type: 'pie',
                marker: { colors: ['#1565C0', '#1976D2', '#1E88E5', '#42A5F5', '#64B5F6'] }
            }], { margin: { t: 0 } });
        }

        console.log('All charts rendered successfully!');
    } catch (error) {
        console.error('Error loading analytics:', error);
        alert('Failed to load analytics data. Please check the console for details.');
    }
}

// Schema Functionality
async function loadSchema() {
    console.log('Loading schema...');
    const container = document.getElementById('schema-container');
    if (!container) {
        console.error('Schema container not found');
        return;
    }

    try {
        const response = await fetch('/api/schema-info');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const schema = await response.json();
        console.log('Schema received:', schema);
        
        container.innerHTML = '';
        
        for (const [table, columns] of Object.entries(schema)) {
            let colHtml = columns.map(c => `
                <li>
                    <span>${c.name}</span>
                    <span class="type-badge">${c.type}</span>
                </li>
            `).join('');
            
            container.innerHTML += `
                <div class="table-card">
                    <div class="table-header">${table.toUpperCase()}</div>
                    <ul class="column-list">${colHtml}</ul>
                </div>
            `;
        }
        console.log('Schema loaded successfully!');
    } catch (error) {
        console.error('Error loading schema:', error);
        container.innerHTML = '<div style="color: red; padding: 2rem; text-align: center;">Error loading schema. Please check console for details.</div>';
    }
}

// Query Runner Functionality
async function loadTables() {
    console.log('Loading tables...');
    const select = document.getElementById('table-select');
    if (!select) {
        console.error('Table select element not found');
        return;
    }

    try {
        const response = await fetch('/api/query-runner/tables');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const tables = await response.json();
        console.log('Tables received:', tables);
        
        tables.forEach(t => {
            const option = document.createElement('option');
            option.value = t;
            option.textContent = t;
            select.appendChild(option);
        });
        console.log('Tables loaded successfully!');
    } catch (error) {
        console.error('Error loading tables:', error);
        alert('Failed to load tables. Please check console for details.');
    }
}

async function loadColumns() {
    const table = document.getElementById('table-select').value;
    const container = document.getElementById('columns-container');
    container.innerHTML = '';
    
    if (!table) return;

    console.log('Loading columns for table:', table);
    try {
        const response = await fetch(`/api/query-runner/columns/${table}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const columns = await response.json();
        console.log('Columns received:', columns);
        
        columns.forEach(c => {
            container.innerHTML += `
                <label>
                    <input type="checkbox" name="columns" value="${c}" checked> ${c}
                </label>
            `;
        });
        console.log('Columns loaded successfully!');
    } catch (error) {
        console.error('Error loading columns:', error);
        alert('Failed to load columns. Please check console for details.');
    }
}

function addFilter() {
    const container = document.getElementById('filters-container');
    const table = document.getElementById('table-select').value;
    
    if (!table) {
        alert('Please select a table first');
        return;
    }

    // We need columns for the dropdown. 
    // For simplicity, we'll fetch them again or assume they are loaded.
    // Let's just use a text input for column name for now to keep it simple, 
    // or better, clone the column list logic.
    // To make it robust, let's just add a row with inputs.
    
    const div = document.createElement('div');
    div.className = 'filter-row';
    div.innerHTML = `
        <input type="text" placeholder="Column Name" class="filter-col">
        <select class="filter-op">
            <option value="=">=</option>
            <option value=">">></option>
            <option value="<"><</option>
            <option value="LIKE">LIKE</option>
        </select>
        <input type="text" placeholder="Value" class="filter-val">
        <button class="btn btn-small" onclick="this.parentElement.remove()">❌</button>
    `;
    container.appendChild(div);
}

async function executeQuery() {
    console.log('Executing query...');
    const table = document.getElementById('table-select').value;
    if (!table) {
        alert('Please select a table first');
        return;
    }

    const checkboxes = document.querySelectorAll('input[name="columns"]:checked');
    const columns = Array.from(checkboxes).map(cb => cb.value);
    
    if (columns.length === 0) {
        alert('Please select at least one column');
        return;
    }

    const filters = [];
    document.querySelectorAll('.filter-row').forEach(row => {
        const col = row.querySelector('.filter-col').value;
        const op = row.querySelector('.filter-op').value;
        const val = row.querySelector('.filter-val').value;
        if (col && val) {
            filters.push({ column: col, operator: op, value: val });
        }
    });

    console.log('Query params:', { table, columns, filters });

    const resultsDiv = document.getElementById('query-results');
    resultsDiv.innerHTML = '<div class="loading" style="padding: 2rem; text-align: center;">Running query...</div>';

    try {
        const response = await fetch('/api/query-runner/execute', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ table, columns, filters })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Query results:', data);
        
        if (data.error) {
            resultsDiv.innerHTML = `<div class="error">Error: ${data.error}</div>`;
            return;
        }

        if (data.rows.length === 0) {
            resultsDiv.innerHTML = '<div>No results found.</div>';
            return;
        }

        let html = '<table><thead><tr>';
        data.columns.forEach(c => html += `<th>${c}</th>`);
        html += '</tr></thead><tbody>';
        
        data.rows.forEach(row => {
            html += '<tr>';
            data.columns.forEach(c => html += `<td>${row[c]}</td>`);
            html += '</tr>';
        });
        html += '</tbody></table>';
        
        resultsDiv.innerHTML = html;
        console.log('Query executed successfully!');

    } catch (error) {
        console.error('Error executing query:', error);
        resultsDiv.innerHTML = `<div style="color: red; padding: 2rem;">Error: ${error.message}</div>`;
    }
}
