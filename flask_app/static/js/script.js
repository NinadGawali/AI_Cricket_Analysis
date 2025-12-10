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
            <div class="message-content">Thinking... analyzing across ODI, T20, Test & IPL data...</div>
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
                    <div class="message-content">Error: ${data.error}</div>
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

// Analytics Functionality - All Formats
async function loadAllAnalytics() {
    // Update tab state
    document.querySelectorAll('.format-tab').forEach(tab => tab.classList.remove('active'));
    document.querySelector('.format-tab:first-child').classList.add('active');
    
    // Show all formats dashboard, hide format-specific
    document.getElementById('all-formats-dashboard').style.display = 'block';
    document.getElementById('format-dashboard').style.display = 'none';

    console.log('Loading all formats analytics data...');
    try {
        const response = await fetch('/api/analytics-data');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        console.log('Analytics data received:', data);

        if (!data || Object.keys(data).length === 0) {
            console.error('No data received from API');
            return;
        }

        const colors = {
            primary: '#1565C0',
            secondary: '#1976D2',
            accent: '#42A5F5',
            light: '#90CAF9',
            dark: '#0D47A1'
        };

        // 1. Wins by Team (Bar)
        if (data.wins_by_team && data.wins_by_team.length > 0) {
            Plotly.newPlot('chart-wins', [{
                x: data.wins_by_team.map(d => d.Wins),
                y: data.wins_by_team.map(d => d.Team),
                type: 'bar',
                orientation: 'h',
                marker: { color: colors.primary }
            }], { margin: { t: 10, l: 150, r: 20, b: 40 } });
        }

        // 2. Matches by Type (Pie)
        if (data.matches_by_type && data.matches_by_type.length > 0) {
            Plotly.newPlot('chart-type', [{
                values: data.matches_by_type.map(d => d.Matches),
                labels: data.matches_by_type.map(d => d.Type),
                type: 'pie',
                marker: { colors: [colors.primary, colors.secondary, colors.accent, colors.dark] }
            }], { margin: { t: 10, b: 10 } });
        }

        // 3. Toss Decision (Pie)
        if (data.toss_decision && data.toss_decision.length > 0) {
            Plotly.newPlot('chart-toss', [{
                values: data.toss_decision.map(d => d.Count),
                labels: data.toss_decision.map(d => d.Decision),
                type: 'pie',
                marker: { colors: [colors.secondary, colors.light] }
            }], { margin: { t: 10, b: 10 } });
        }

        // 4. Win Method (Pie)
        if (data.win_method && data.win_method.length > 0) {
            Plotly.newPlot('chart-method', [{
                values: data.win_method.map(d => d.Count),
                labels: data.win_method.map(d => d.Method),
                type: 'pie',
                marker: { colors: [colors.primary, colors.accent, colors.light] }
            }], { margin: { t: 10, b: 10 } });
        }

        // 5. Top Scorers (Bar)
        if (data.top_scorers && data.top_scorers.length > 0) {
            Plotly.newPlot('chart-scorers', [{
                x: data.top_scorers.map(d => d.Runs),
                y: data.top_scorers.map(d => d.Player),
                type: 'bar',
                orientation: 'h',
                marker: { color: colors.secondary }
            }], { margin: { t: 10, l: 150, r: 20, b: 40 } });
        }

        // 6. Top Wicket Takers (Bar)
        if (data.top_wicket_takers && data.top_wicket_takers.length > 0) {
            Plotly.newPlot('chart-wickets', [{
                x: data.top_wicket_takers.map(d => d.Wickets),
                y: data.top_wicket_takers.map(d => d.Player),
                type: 'bar',
                orientation: 'h',
                marker: { color: colors.dark }
            }], { margin: { t: 10, l: 150, r: 20, b: 40 } });
        }

        // 7. Player of Match Awards (Bar)
        if (data.player_of_match && data.player_of_match.length > 0) {
            Plotly.newPlot('chart-pom', [{
                x: data.player_of_match.map(d => d.Awards),
                y: data.player_of_match.map(d => d.Player),
                type: 'bar',
                orientation: 'h',
                marker: { color: colors.accent }
            }], { margin: { t: 10, l: 150, r: 20, b: 40 } });
        }

        // 8. Top Fielders (Bar)
        if (data.top_fielders && data.top_fielders.length > 0) {
            Plotly.newPlot('chart-fielders', [{
                x: data.top_fielders.map(d => d.Dismissals),
                y: data.top_fielders.map(d => d.Fielder),
                type: 'bar',
                orientation: 'h',
                marker: { color: colors.primary }
            }], { margin: { t: 10, l: 150, r: 20, b: 40 } });
        }

        // 9. Average Runs per Format (Bar)
        if (data.avg_runs_per_format && data.avg_runs_per_format.length > 0) {
            Plotly.newPlot('chart-avg-runs', [{
                x: data.avg_runs_per_format.map(d => d.Format),
                y: data.avg_runs_per_format.map(d => d.AvgRuns),
                type: 'bar',
                marker: { color: [colors.primary, colors.secondary, colors.accent, colors.dark] }
            }], { margin: { t: 10, b: 40 } });
        }

        // 10. Super Over Matches (Bar)
        if (data.super_over_matches && data.super_over_matches.length > 0) {
            Plotly.newPlot('chart-super-over', [{
                x: data.super_over_matches.map(d => d.Format),
                y: data.super_over_matches.map(d => d.Matches),
                type: 'bar',
                marker: { color: colors.secondary }
            }], { margin: { t: 10, b: 40 } });
        }

        // 11. Dismissal Types (Pie)
        if (data.dismissal_types && data.dismissal_types.length > 0) {
            Plotly.newPlot('chart-dismissal', [{
                values: data.dismissal_types.map(d => d.Count),
                labels: data.dismissal_types.map(d => d.Type),
                type: 'pie'
            }], { margin: { t: 10, b: 10 } });
        }

        // 12. Matches by City (Pie)
        if (data.matches_by_city && data.matches_by_city.length > 0) {
            Plotly.newPlot('chart-city', [{
                values: data.matches_by_city.map(d => d.Matches),
                labels: data.matches_by_city.map(d => d.City),
                type: 'pie'
            }], { margin: { t: 10, b: 10 } });
        }

        // 13. IPL Teams Performance (Bar)
        if (data.ipl_team_performance && data.ipl_team_performance.length > 0) {
            Plotly.newPlot('chart-ipl-teams', [{
                x: data.ipl_team_performance.map(d => d.Team),
                y: data.ipl_team_performance.map(d => d.WinPct),
                type: 'bar',
                marker: { color: colors.primary },
                text: data.ipl_team_performance.map(d => `${d.Wins}/${d.Matches} wins`),
                textposition: 'outside'
            }], { 
                margin: { t: 30, b: 100, l: 50, r: 20 },
                xaxis: { tickangle: -45 },
                yaxis: { title: 'Win %' }
            });
        }

        // 14. Matches per Season (Line)
        if (data.matches_per_season && data.matches_per_season.length > 0) {
            Plotly.newPlot('chart-season', [{
                x: data.matches_per_season.map(d => d.Season),
                y: data.matches_per_season.map(d => d.Matches),
                type: 'scatter',
                mode: 'lines+markers',
                line: { color: colors.primary, width: 3 },
                marker: { size: 8 }
            }], { margin: { t: 20, b: 40 } });
        }

        console.log('All charts rendered successfully!');
    } catch (error) {
        console.error('Error loading analytics:', error);
    }
}

// Format-specific Analytics
async function loadFormatStats(format) {
    // Update tab state
    document.querySelectorAll('.format-tab').forEach(tab => tab.classList.remove('active'));
    event.target.classList.add('active');
    
    // Show format dashboard, hide all formats
    document.getElementById('all-formats-dashboard').style.display = 'none';
    document.getElementById('format-dashboard').style.display = 'block';
    
    const formatNames = {
        'odi': 'ODI',
        't20': 'T20 International',
        'test': 'Test',
        'ipl': 'IPL'
    };
    document.getElementById('format-title').textContent = `${formatNames[format]} Statistics`;

    console.log(`Loading ${format} format statistics...`);
    try {
        const response = await fetch(`/api/format-stats/${format}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        console.log(`${format} data received:`, data);

        const colors = {
            primary: '#1565C0',
            secondary: '#1976D2',
            accent: '#42A5F5',
            dark: '#0D47A1'
        };

        // Top Scorers
        if (data.top_scorers && data.top_scorers.length > 0) {
            Plotly.newPlot('format-chart-scorers', [{
                x: data.top_scorers.map(d => d.Runs),
                y: data.top_scorers.map(d => d.Player),
                type: 'bar',
                orientation: 'h',
                marker: { color: colors.primary },
                text: data.top_scorers.map(d => `${d.Matches} matches`),
                textposition: 'outside'
            }], { margin: { t: 10, l: 150, r: 80, b: 40 } });
        }

        // Top Wicket Takers
        if (data.top_wicket_takers && data.top_wicket_takers.length > 0) {
            Plotly.newPlot('format-chart-wickets', [{
                x: data.top_wicket_takers.map(d => d.Wickets),
                y: data.top_wicket_takers.map(d => d.Player),
                type: 'bar',
                orientation: 'h',
                marker: { color: colors.secondary }
            }], { margin: { t: 10, l: 150, r: 20, b: 40 } });
        }

        // Wins by Team
        if (data.wins_by_team && data.wins_by_team.length > 0) {
            Plotly.newPlot('format-chart-wins', [{
                x: data.wins_by_team.map(d => d.Wins),
                y: data.wins_by_team.map(d => d.Team),
                type: 'bar',
                orientation: 'h',
                marker: { color: colors.accent }
            }], { margin: { t: 10, l: 150, r: 20, b: 40 } });
        }

        // Top Venues
        if (data.top_venues && data.top_venues.length > 0) {
            Plotly.newPlot('format-chart-venues', [{
                x: data.top_venues.map(d => d.Matches),
                y: data.top_venues.map(d => d.Venue.substring(0, 30)),
                type: 'bar',
                orientation: 'h',
                marker: { color: colors.dark }
            }], { margin: { t: 10, l: 180, r: 20, b: 40 } });
        }

        console.log(`${format} charts rendered successfully!`);
    } catch (error) {
        console.error(`Error loading ${format} stats:`, error);
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
        
        // Group tables by format
        const formats = {
            'ODI Tables': [],
            'T20 Tables': [],
            'Test Tables': [],
            'IPL Tables': []
        };
        
        for (const [table, columns] of Object.entries(schema)) {
            if (table.startsWith('odi_')) formats['ODI Tables'].push({table, columns});
            else if (table.startsWith('t20_')) formats['T20 Tables'].push({table, columns});
            else if (table.startsWith('test_')) formats['Test Tables'].push({table, columns});
            else if (table.startsWith('ipl_')) formats['IPL Tables'].push({table, columns});
        }
        
        for (const [format, tables] of Object.entries(formats)) {
            if (tables.length === 0) continue;
            
            container.innerHTML += `<h3 class="format-header">${format}</h3>`;
            let tableHtml = '<div class="schema-grid">';
            
            tables.forEach(({table, columns}) => {
                let colHtml = columns.map(c => `
                    <li>
                        <span>${c.name}</span>
                        <span class="type-badge">${c.type}</span>
                    </li>
                `).join('');
                
                tableHtml += `
                    <div class="table-card">
                        <div class="table-header">${table}</div>
                        <ul class="column-list">${colHtml}</ul>
                    </div>
                `;
            });
            
            tableHtml += '</div>';
            container.innerHTML += tableHtml;
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
        
        // Group by format
        const groups = {
            'ODI': tables.filter(t => t.startsWith('odi_')),
            'T20': tables.filter(t => t.startsWith('t20_')),
            'Test': tables.filter(t => t.startsWith('test_')),
            'IPL': tables.filter(t => t.startsWith('ipl_'))
        };
        
        for (const [format, formatTables] of Object.entries(groups)) {
            const optgroup = document.createElement('optgroup');
            optgroup.label = format;
            formatTables.forEach(t => {
                const option = document.createElement('option');
                option.value = t;
                option.textContent = t;
                optgroup.appendChild(option);
            });
            select.appendChild(optgroup);
        }
        console.log('Tables loaded successfully!');
    } catch (error) {
        console.error('Error loading tables:', error);
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
    }
}

function addFilter() {
    const container = document.getElementById('filters-container');
    const table = document.getElementById('table-select').value;
    
    if (!table) {
        alert('Please select a table first');
        return;
    }

    const div = document.createElement('div');
    div.className = 'filter-row';
    div.innerHTML = `
        <input type="text" placeholder="Column Name" class="filter-col">
        <select class="filter-op">
            <option value="=">=</option>
            <option value=">">></option>
            <option value="<"><</option>
            <option value=">=">>=</option>
            <option value="<="><=</option>
            <option value="LIKE">LIKE</option>
            <option value="!=">!=</option>
        </select>
        <input type="text" placeholder="Value" class="filter-val">
        <button class="btn btn-small" onclick="this.parentElement.remove()">Remove</button>
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

        let html = `<p><strong>${data.rows.length}</strong> rows returned</p>`;
        html += '<table><thead><tr>';
        data.columns.forEach(c => html += `<th>${c}</th>`);
        html += '</tr></thead><tbody>';
        
        data.rows.forEach(row => {
            html += '<tr>';
            data.columns.forEach(c => html += `<td>${row[c] !== null ? row[c] : ''}</td>`);
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
