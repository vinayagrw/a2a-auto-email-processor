// DOM Elements
const collectionsList = document.getElementById('collections-list');
const collectionContent = document.getElementById('collection-content');
const collectionTitle = document.getElementById('collection-title');
const loadingSpinner = document.getElementById('loading-spinner');
const loadingText = document.getElementById('loading-text');
const errorAlert = document.getElementById('error-alert');
const errorMessage = document.getElementById('error-message');
const prevPageBtn = document.getElementById('prev-page');
const nextPageBtn = document.getElementById('next-page');
const pageInfo = document.getElementById('page-info');

// API Configuration - Use relative URL
const API_BASE_URL = '';  // Relative to the current domain

// State
let currentCollection = null;
let currentPage = 0;
const limit = 10;
let totalItems = 0;

// Helper function to make API requests
async function apiRequest(endpoint, options = {}) {
    // Ensure endpoint starts with a slash
    if (!endpoint.startsWith('/')) {
        endpoint = '/' + endpoint;
    }
    
    // Remove any double slashes that might occur from joining base URL and endpoint
    const url = `${API_BASE_URL}${endpoint}`.replace(/([^:]\/)\/+/g, '$1');
    
    console.log(`[API] Request: ${options.method || 'GET'} ${url}`, options.body ? { body: options.body } : '');
    
    try {
        // Add default headers if not provided
        const headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            ...(options.headers || {})
        };
        
        // Create the request options
        const requestOptions = {
            ...options,
            headers,
            credentials: 'same-origin', // Use 'same-origin' instead of 'include' for better security
            mode: 'cors' // Explicitly set CORS mode
        };
        
        // Make the request
        const response = await fetch(url, requestOptions);
        
        console.log(`[API] Response: ${response.status} ${response.statusText}`, { url });
        
        // Handle non-2xx responses
        if (!response.ok) {
            let errorMessage = `HTTP error! status: ${response.status}`;
            let errorDetails = '';
            
            // Try to get error details from response
            try {
                const errorData = await response.json();
                errorDetails = errorData.detail || JSON.stringify(errorData);
            } catch (e) {
                try {
                    errorDetails = await response.text();
                } catch (e) {
                    errorDetails = 'No additional error information available';
                }
            }
            
            console.error('[API] Error details:', errorDetails);
            
            // Create a more detailed error
            const error = new Error(errorMessage);
            error.status = response.status;
            error.details = errorDetails;
            throw error;
        }
        
        // Parse and return the response
        try {
            return await response.json();
        } catch (e) {
            console.error('[API] Failed to parse JSON response:', e);
            throw new Error('Failed to parse response as JSON');
        }
        
    } catch (error) {
        console.error('[API] Request failed:', {
            url,
            error: error.message,
            stack: error.stack,
            ...(error.details && { details: error.details })
        });
        
        // Re-throw with a more user-friendly message if needed
        if (error.name === 'TypeError' && error.message.includes('fetch')) {
            throw new Error('Failed to connect to the server. Please check your connection and try again.');
        }
        
        throw error;
    }
}

// Show or hide loading state
function showLoading(show, message = '') {
    console.log(`Loading: ${show ? 'true' : 'false'}`, message);
    if (loadingSpinner) loadingSpinner.style.display = show ? 'block' : 'none';
    if (loadingText) loadingText.textContent = message || 'Loading...';
    
    // Disable buttons while loading
    const buttons = document.querySelectorAll('button');
    buttons.forEach(btn => {
        if (btn.id !== 'retry-button') {
            btn.disabled = show;
        }
    });
}

// Show error message
function showError(message) {
    console.error('Showing error:', message);
    if (errorMessage) errorMessage.textContent = message;
    if (errorAlert) errorAlert.style.display = 'block';
    
    // Auto-hide error after 10 seconds
    setTimeout(() => {
        if (errorAlert) errorAlert.style.display = 'none';
    }, 10000);
}

// Show message when no collections are found
function showNoCollectionsMessage() {
    console.log('No collections found');
    if (collectionContent) {
        collectionContent.innerHTML = `
            <div class="alert alert-info">
                No collections found in the database. Please add some data to ChromaDB.
            </div>
        `;
    }
    if (collectionTitle) {
        collectionTitle.textContent = 'No Collections';
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM fully loaded, initializing...');
    
    // Set up event listeners for pagination
    if (prevPageBtn) {
        prevPageBtn.addEventListener('click', () => {
            if (currentPage > 0) {
                currentPage--;
                loadCollectionData();
            }
        });
    }
    
    if (nextPageBtn) {
        nextPageBtn.addEventListener('click', () => {
            const maxPage = Math.ceil(totalItems / limit) - 1;
            if (currentPage < maxPage) {
                currentPage++;
                loadCollectionData();
            }
        });
    }
    
    // Add retry button handler if it exists
    const retryButton = document.getElementById('retry-button');
    if (retryButton) {
        retryButton.addEventListener('click', () => {
            if (currentCollection) {
                loadCollectionData();
            } else {
                loadCollections();
            }
        });
    }
    
    // Load collections when the page loads
    loadCollections();
});

// Load all collections
async function loadCollections() {
    console.log('[Collections] Loading collections...');
    showLoading(true, 'Loading collections...');
    
    try {
        // Clear any previous error
        if (errorAlert) errorAlert.style.display = 'none';
        
        // Make the API request
        const collections = await apiRequest('/api/collections');
        console.log('[Collections] Loaded successfully:', collections);
        
        // Handle the response
        if (Array.isArray(collections) && collections.length > 0) {
            renderCollectionsList(collections);
            
            // If we have a current collection, refresh its data
            if (currentCollection) {
                await loadCollectionData();
            }
        } else {
            console.log('[Collections] No collections found');
            showNoCollectionsMessage();
        }
        
        return collections;
        
    } catch (error) {
        console.error('[Collections] Failed to load collections:', {
            error: error.message,
            status: error.status,
            details: error.details
        });
        
        // Show a user-friendly error message
        let errorMessage = 'Failed to load collections';
        
        if (error.status === 404) {
            errorMessage = 'API endpoint not found. The server might be misconfigured.';
        } else if (error.status === 500) {
            errorMessage = 'Server error. Please try again later.';
        } else if (error.message.includes('Failed to fetch')) {
            errorMessage = 'Could not connect to the server. Please check your connection.';
        } else if (error.details) {
            errorMessage += `: ${error.details}`;
        } else {
            errorMessage += `: ${error.message}`;
        }
        
        showError(errorMessage);
        
        // Re-throw to allow callers to handle the error if needed
        throw error;
    } finally {
        showLoading(false);
    }
}

// Helper function to escape HTML
function escapeHtml(unsafe) {
    if (typeof unsafe !== 'string') return unsafe;
    return unsafe
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

// Render the collections list
function renderCollectionsList(collections) {
    console.log('[Collections] Rendering collections list:', collections);
    
    if (!collectionsList) {
        console.error('[Collections] Error: Collections list element not found in the DOM');
        return;
    }
    
    try {
        // Clear the current list
        collectionsList.innerHTML = '';
        
        // Handle empty collections
        if (!Array.isArray(collections) || collections.length === 0) {
            console.log('[Collections] No collections to display');
            collectionsList.innerHTML = `
                <div class="alert alert-info m-2">
                    <i class="fas fa-info-circle me-2"></i>
                    No collections found in the database.
                </div>
            `;
            return;
        }
        
        // Create a document fragment for better performance
        const fragment = document.createDocumentFragment();
        let hasValidCollections = false;
        
        // Process each collection
        collections.forEach((collection, index) => {
            try {
                // Validate collection data
                if (!collection || typeof collection !== 'object' || !collection.name) {
                    console.warn(`[Collections] Invalid collection data at index ${index}:`, collection);
                    return;
                }
                
                const { name, count = 0, metadata = {} } = collection;
                
                // Create collection item
                const item = document.createElement('div');
                item.className = `collection-item list-group-item list-group-item-action d-flex justify-content-between align-items-center`;
                
                // Add active class if this is the current collection
                if (currentCollection === name) {
                    item.classList.add('active');
                }
                
                // Set item content
                item.innerHTML = `
                    <div class="d-flex align-items-center">
                        <i class="fas fa-folder me-2"></i>
                        <span class="collection-name text-truncate" title="${escapeHtml(name)}">
                            ${escapeHtml(name)}
                        </span>
                    </div>
                    <span class="badge bg-primary rounded-pill">${count}</span>
                `;
                
                // Add click handler
                item.addEventListener('click', async (e) => {
                    e.preventDefault();
                    
                    try {
                        // Update active state
                        document.querySelectorAll('.collection-item').forEach(el => {
                            el.classList.remove('active');
                        });
                        item.classList.add('active');
                        
                        console.log(`[Collections] Selected collection: ${name}`);
                        await selectCollection(name);
                    } catch (error) {
                        console.error(`[Collections] Error selecting collection ${name}:`, error);
                        showError(`Failed to select collection: ${error.message}`);
                    }
                });
                
                // Add to fragment
                fragment.appendChild(item);
                hasValidCollections = true;
                
            } catch (error) {
                console.error(`[Collections] Error rendering collection at index ${index}:`, error);
            }
        });
        
        // Add all items to the DOM at once
        if (hasValidCollections) {
            collectionsList.appendChild(fragment);
            
            // If no collection is selected, select the first one
            if (!currentCollection && collections.length > 0) {
                const firstValidCollection = collections.find(c => c && c.name);
                if (firstValidCollection) {
                    console.log(`[Collections] Auto-selecting first collection: ${firstValidCollection.name}`);
                    selectCollection(firstValidCollection.name);
                }
            }
        } else {
            collectionsList.innerHTML = `
                <div class="alert alert-warning m-2">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    No valid collections found.
                </div>
            `;
        }
        
    } catch (error) {
        console.error('[Collections] Error rendering collections list:', error);
        collectionsList.innerHTML = `
            <div class="alert alert-danger m-2">
                <i class="fas fa-exclamation-circle me-2"></i>
                Error displaying collections: ${escapeHtml(error.message)}
            </div>
        `;
    }
}

// Select a collection and load its data
async function selectCollection(collectionName) {
    if (!collectionName) {
        console.error('[Collections] Error: No collection name provided');
        return;
    }
    
    console.log(`[Collections] Selecting collection: ${collectionName}`);
    showLoading(true, `Loading ${collectionName}...`);
    
    try {
        // Update the current collection
        const previousCollection = currentCollection;
        currentCollection = collectionName;
        currentPage = 0; // Reset to first page
        
        // Update the active state in the UI
        try {
            document.querySelectorAll('.collection-item').forEach(item => {
                const nameSpan = item.querySelector('.collection-name');
                if (nameSpan && nameSpan.textContent === collectionName) {
                    item.classList.add('active');
                } else {
                    item.classList.remove('active');
                }
            });
        } catch (uiError) {
            console.warn('[UI] Error updating collection selection UI:', uiError);
            // Continue even if UI update fails
        }
        
        // Update the collection title
        if (collectionTitle) {
            collectionTitle.textContent = `Loading ${collectionName}...`;
        }
        
        // Load the collection data
        await loadCollectionData();
        
        console.log(`[Collections] Successfully loaded collection: ${collectionName}`);
        
    } catch (error) {
        console.error(`[Collections] Error selecting collection ${collectionName}:`, {
            error: error.message,
            stack: error.stack,
            ...(error.details && { details: error.details })
        });
        
        // Revert to previous collection if available
        if (previousCollection && previousCollection !== collectionName) {
            console.log(`[Collections] Reverting to previous collection: ${previousCollection}`);
            currentCollection = previousCollection;
        } else {
            currentCollection = null;
        }
        
        // Show error to user
        let errorMessage = `Failed to load collection ${collectionName}`;
        if (error.status === 404) {
            errorMessage = `Collection not found: ${collectionName}`;
        } else if (error.message.includes('Failed to fetch')) {
            errorMessage = 'Could not connect to the server. Please check your connection.';
        } else if (error.details) {
            errorMessage += `: ${error.details}`;
        } else {
            errorMessage += `: ${error.message}`;
        }
        
        showError(errorMessage);
        
        // Re-throw to allow callers to handle the error
        throw error;
        
    } finally {
        showLoading(false);
    }
}

// Load data for the current collection
async function loadCollectionData() {
    if (!currentCollection) {
        const errorMsg = 'No collection selected';
        console.error(`[Collection] ${errorMsg}`);
        showError(errorMsg);
        return;
    }
    
    const collectionName = currentCollection; // Capture the current collection name in case it changes
    const currentRequestPage = currentPage; // Capture the current page in case it changes
    
    console.log(`[Collection] Loading data for collection: ${collectionName}, page: ${currentRequestPage}`);
    showLoading(true, `Loading documents from ${collectionName}...`);
    
    try {
        // Clear any previous errors
        if (errorAlert) errorAlert.style.display = 'none';
        
        // Build the API URL with proper encoding
        const endpoint = `/api/collections/${encodeURIComponent(collectionName)}`;
        const params = new URLSearchParams({
            limit: limit.toString(),
            offset: (currentRequestPage * limit).toString()
        });
        
        const url = `${endpoint}?${params}`;
        console.log(`[Collection] Fetching: ${url}`);
        
        // Make the API request
        const data = await apiRequest(url);
        console.log(`[Collection] Received data for ${collectionName}:`, {
            total: data.total,
            count: data.data ? data.data.length : 0,
            page: currentRequestPage
        });
        
        // Check if the collection is still the same
        if (currentCollection !== collectionName) {
            console.log(`[Collection] Collection changed from ${collectionName} to ${currentCollection}, aborting update`);
            return;
        }
        
        // Validate the response
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Update the total items count
        const itemCount = data.total || 0;
        totalItems = itemCount;
        
        // Update the UI
        if (collectionTitle) {
            collectionTitle.textContent = `${collectionName} (${itemCount} document${itemCount !== 1 ? 's' : ''})`;
        }
        
        // Render the documents or show a message if there are none
        if (data.data && Array.isArray(data.data)) {
            if (data.data.length > 0) {
                renderDocuments(data.data);
            } else {
                console.log(`[Collection] No documents found in collection: ${collectionName}`);
                renderDocuments([]);
                
                if (collectionContent) {
                    collectionContent.innerHTML = `
                        <div class="alert alert-info">
                            <i class="fas fa-info-circle me-2"></i>
                            No documents found in this collection.
                        </div>
                    `;
                }
            }
        } else {
            console.warn('[Collection] Invalid data format in response:', data);
            throw new Error('Invalid data format received from server');
        }
        
        // Update pagination controls
        updatePagination();
        
    } catch (error) {
        console.error(`[Collection] Error loading data for ${collectionName}:`, {
            error: error.message,
            status: error.status,
            details: error.details,
            stack: error.stack
        });
        
        // Only show the error if we're still on the same collection
        if (currentCollection === collectionName) {
            let errorMessage = `Failed to load documents from ${collectionName}`;
            
            if (error.status === 404) {
                errorMessage = `Collection not found: ${collectionName}`;
            } else if (error.status === 500) {
                errorMessage = 'Server error while loading documents';
            } else if (error.message.includes('Failed to fetch')) {
                errorMessage = 'Could not connect to the server. Please check your connection.';
            } else if (error.details) {
                errorMessage += `: ${error.details}`;
            } else {
                errorMessage += `: ${error.message}`;
            }
            
            showError(errorMessage);
            
            // Clear the content area
            if (collectionContent) {
                collectionContent.innerHTML = '';
            }
        }
        
        // Re-throw to allow callers to handle the error
        throw error;
        
    } finally {
        // Only update loading state if we're still working with the same collection
        if (currentCollection === collectionName) {
            showLoading(false);
        }
    }
}

// Render documents in the collection
function renderDocuments(documents) {
    console.log(`[Documents] Rendering ${documents ? documents.length : 0} documents`);
    
    if (!collectionContent) {
        console.error('[Documents] Error: Collection content element not found');
        return;
    }
    
    // Clear any previous content
    collectionContent.innerHTML = '';
    
    // Handle empty documents array
    if (!Array.isArray(documents) || documents.length === 0) {
        console.log('[Documents] No documents to display');
        collectionContent.innerHTML = `
            <div class="alert alert-info">
                <i class="fas fa-info-circle me-2"></i>
                No documents found in this collection.
            </div>
        `;
        return;
    }
    
    // Create a document fragment for better performance
    const fragment = document.createDocumentFragment();
    let hasValidDocuments = false;
    
    // Create a container for the documents
    const container = document.createElement('div');
    container.className = 'documents-container';
    
    // Add a header with document count
    const header = document.createElement('div');
    header.className = 'documents-header mb-3';
    header.innerHTML = `
        <div class="d-flex justify-content-between align-items-center">
            <h5 class="mb-0">
                <i class="fas fa-file-alt me-2"></i>
                ${documents.length} document${documents.length !== 1 ? 's' : ''} found
            </h5>
            <div class="text-muted small">
                Showing ${documents.length} of ${totalItems}
            </div>
        </div>
    `;
    container.appendChild(header);
    
    // Create a row for the document cards
    const row = document.createElement('div');
    row.className = 'row g-4';
    container.appendChild(row);
    
    try {
        documents.forEach((doc, index) => {
            try {
                if (!doc) {
                    console.warn('Undefined document at index:', index);
                    return;
                        <div class="alert alert-warning mb-3">
                            <strong>Warning:</strong> Document is undefined or null
                        </div>
                    `;
                }
                
                // Safely get document ID
                const docId = doc.id || `document-${index}`;
                
                // Format the document content
                let content = '';
                if (doc.document !== undefined && doc.document !== null) {
                    if (typeof doc.document === 'object') {
                        content = JSON.stringify(doc.document, null, 2);
                    } else {
                        content = String(doc.document);
                    }
                } else {
                    content = '[No content]';
                }
                
                // Format metadata if it exists
                let metadataHtml = '';
                if (doc.metadata && typeof doc.metadata === 'object' && Object.keys(doc.metadata).length > 0) {
                    const metadataItems = Object.entries(doc.metadata)
                        .map(([key, value]) => {
                            let displayValue;
                            try {
                                displayValue = typeof value === 'object' 
                                    ? JSON.stringify(value, null, 2) 
                                    : String(value);
                            } catch (e) {
                                console.warn(`Error formatting metadata value for key '${key}':`, e);
                                displayValue = '[Error formatting value]';
                            }
                            
                            return `
                                <div class="metadata-item">
                                    <span class="metadata-key">${escapeHtml(key)}:</span>
                                    <span class="metadata-value">${escapeHtml(displayValue)}</span>
                                </div>
                            `;
                        })
                        .join('');
                    
                    metadataHtml = `
                        <div class="mt-3">
                            <h6 class="card-subtitle mb-2 text-muted">Metadata</h6>
                            <div class="metadata p-2 rounded">${metadataItems}</div>
                        </div>
                    `;
                }
                
                return `
                    <div class="card document-card mb-4">
                        <div class="card-header d-flex justify-content-between align-items-center">
                            <span>${escapeHtml(docId)}</span>
                            <button class="btn btn-sm btn-outline-secondary copy-btn" 
                                    data-content="${escapeHtml(JSON.stringify(doc, null, 2))}"
                                    title="Copy to clipboard">
                                <i class="far fa-copy"></i>
                            </button>
                        </div>
                        <div class="card-body">
                            <h6 class="card-subtitle mb-2 text-muted">Document Content</h6>
                            <div class="document-content p-2 rounded">${escapeHtml(content)}</div>
                            ${metadataHtml}
                        </div>
                    </div>
                `;
            } catch (error) {
                console.error('Error rendering document at index', index, ':', error);
                return `
                    <div class="alert alert-danger mb-3">
                        <strong>Error rendering document:</strong> ${escapeHtml(error.message)}
                        <button class="btn btn-sm btn-outline-dark ms-2 copy-btn" 
                                data-content="${escapeHtml(JSON.stringify(doc, null, 2))}"
                                title="Copy to clipboard">
                            <i class="far fa-copy"></i> Copy Raw Data
                        </button>
                    </div>
                `;
            }
        }).join('');
        
        // Add event listeners to copy buttons
        document.querySelectorAll('.copy-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                const content = btn.getAttribute('data-content');
                if (content) {
                    navigator.clipboard.writeText(content).then(() => {
                        const originalText = btn.innerHTML;
                        btn.innerHTML = '<i class="fas fa-check"></i> Copied!';
                        btn.classList.add('text-success');
                        setTimeout(() => {
                            btn.innerHTML = originalText;
                            btn.classList.remove('text-success');
                        }, 2000);
                    }).catch(err => {
                        console.error('Failed to copy:', err);
                    });
                }
            });
        });
        
    } catch (error) {
        console.error('Error rendering documents:', error);
        collectionContent.innerHTML = `
            <div class="alert alert-danger">
                <strong>Error rendering documents:</strong> ${escapeHtml(error.message)}
            </div>
        `;
    }
}

// Update pagination controls
function updatePagination() {
    prevPageBtn.disabled = currentPage === 0;
    nextPageBtn.disabled = (currentPage + 1) * limit >= totalItems;
    
    const pageInfo = document.getElementById('page-info');
    if (pageInfo) {
        const start = (currentPage * limit) + 1;
        const end = Math.min((currentPage + 1) * limit, totalItems);
        pageInfo.textContent = `Showing ${start}-${end} of ${totalItems}`;
    }
}

// Show loading state
function showLoading(isLoading, message = 'Loading...') {
    if (isLoading) {
        // Create loading indicator if it doesn't exist
        if (!loadingIndicator) {
            const indicator = document.createElement('div');
            indicator.id = 'loading-indicator';
            indicator.className = 'loading-indicator';
            document.body.appendChild(indicator);
        }
        
        // Show loading message in content area if specified
        if (message && collectionContent) {
            collectionContent.innerHTML = `
                <div class="d-flex align-items-center justify-content-center" style="height: 200px;">
                    <div class="spinner-border text-primary" role="status"></div>
                    <span class="ms-2">${message}</span>
                </div>
            `;
        }
    } else {
        // Hide loading indicator
        if (loadingIndicator) {
            loadingIndicator.style.display = 'none';
        }
    }
}

// Show error message
function showError(message) {
    collectionContent.innerHTML = `
        <div class="alert alert-danger">
            <strong>Error:</strong> ${escapeHtml(message)}
        </div>
    `;
}

// Show no collections message
function showNoCollectionsMessage() {
    collectionContent.innerHTML = `
        <div class="alert alert-info">
            No collections found in the database. The ChromaDB might be empty or not properly initialized.
        </div>
    `;
}

// Helper function to escape HTML
function escapeHtml(unsafe) {
    if (unsafe === null || unsafe === undefined) return '';
    return unsafe
        .toString()
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}
